import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from preprocessing import train_dataset, test_dataset, collate_fn, train_sample_weights
from setter import Setter, VOCAB_SIZE, PAD_TOKEN_ID
from tqdm import tqdm

# Constants
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-4
GRAD_CLIP_NORM = 1.0
LENGTH_LOSS_WEIGHT = 0.1  # weight of the auxiliary length-prediction loss vs. the main CE loss
SEED = 42
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train():
    set_seed(SEED)

    # Weighted so rare (grade, angle) combinations are seen roughly as often as common
    # ones, instead of being drowned out by dominant buckets like angle=40.
    sampler = WeightedRandomSampler(
        train_sample_weights, num_samples=len(train_dataset), replacement=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn
    )

    model = Setter(vocab_size=VOCAB_SIZE).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(DEVICE)
            grade = batch["grade"].to(DEVICE)
            angle = batch["angle"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            num_holds = batch["num_holds"].to(DEVICE)

            logits = model(input_ids, grade, angle, labels=labels)
            ce_loss = criterion(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1))
            length_loss = nn.functional.mse_loss(model.predict_length(grade, angle), num_holds)
            loss = ce_loss + LENGTH_LOSS_WEIGHT * length_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                grade = batch["grade"].to(DEVICE)
                angle = batch["angle"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                num_holds = batch["num_holds"].to(DEVICE)

                logits = model(input_ids, grade, angle, labels=labels)
                ce_loss = criterion(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1))
                length_loss = nn.functional.mse_loss(model.predict_length(grade, angle), num_holds)
                val_loss += (ce_loss + LENGTH_LOSS_WEIGHT * length_loss).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        # Two kinds of file: lightweight "deploy" checkpoints (weights only - what
        # generate.py and Flask load) and full "resume" checkpoints (add optimizer
        # state + epoch/loss, for continuing training locally). Deploy checkpoints are
        # small enough to commit to git; GitHub hard-rejects any single blob over 100MB,
        # and a full checkpoint with Adam's optimizer state is ~3x the weights-only size
        # - resume checkpoints are gitignored and meant to stay local.
        resume_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "vocab_size": VOCAB_SIZE,
        }
        torch.save(model.state_dict(), "kilter_setter_latest.pt")
        torch.save(resume_checkpoint, "kilter_setter_latest_resume.pt")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "kilter_setter_best.pt")
            torch.save(resume_checkpoint, "kilter_setter_best_resume.pt")
            print(f"  New best val loss ({best_val_loss:.4f}) -> saved kilter_setter_best.pt")


if __name__ == "__main__":
    train()
