import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from preprocessing import train_dataset, test_dataset, collate_fn, train_sample_weights
from setter import Setter, VOCAB_SIZE, PAD_TOKEN_ID, load_checkpoint_state_dict
from evaluate import run_evaluation
from tqdm import tqdm

# Constants
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-4
GRAD_CLIP_NORM = 1.0
LENGTH_LOSS_WEIGHT = 0.1  # weight of the auxiliary length-prediction loss vs. the main CE loss
FOOT_FRACTION_LOSS_WEIGHT = 0.1  # weight of the auxiliary foot-fraction loss vs. the main CE loss
SPACING_LOSS_WEIGHT = 0.1  # weight of the auxiliary spacing-prediction loss vs. the main CE loss
WEIGHT_DECAY = 0.01  # AdamW decoupled weight decay, standard-ish default
LABEL_SMOOTHING = 0.1  # softens the CE target so the model isn't pushed toward total certainty
SEED = 42

# If set and the file exists, model weights are warm-started from it (non-strict, so
# newly added components like coord_embed/foot_fraction_head just keep their fresh
# random init) instead of training from scratch. The optimizer is always fresh - Adam
# momentum from a differently-shaped parameter set can't be meaningfully carried over.
WARM_START_PATH = "kilter_setter_best.pt"
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

    if WARM_START_PATH and os.path.exists(WARM_START_PATH):
        state_dict = torch.load(WARM_START_PATH, map_location=DEVICE, weights_only=True)
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"Warm-started from {WARM_START_PATH}")
            print(f"  new (randomly-initialized) params: {missing}")
            if unexpected:
                print(f"  ignored unexpected checkpoint keys: {unexpected}")
        except RuntimeError as e:
            # strict=False only tolerates missing/extra keys, not shape mismatches -
            # if the architecture itself changed (e.g. d_model/num_layers), every
            # tensor shape differs and there's nothing meaningful to transfer.
            print(f"Could not warm-start from {WARM_START_PATH} (architecture changed): {e}")
            print("Training this model from scratch instead.")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
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
            foot_fraction = batch["foot_fraction"].to(DEVICE)
            avg_spacing = batch["avg_spacing"].to(DEVICE)

            logits = model(input_ids, grade, angle, labels=labels)
            ce_loss = criterion(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1))
            length_loss = nn.functional.mse_loss(model.predict_length(grade, angle), num_holds)
            foot_loss = nn.functional.mse_loss(model.predict_foot_fraction(grade, angle), foot_fraction)
            spacing_loss = nn.functional.mse_loss(model.predict_spacing(grade, angle), avg_spacing)
            loss = (ce_loss + LENGTH_LOSS_WEIGHT * length_loss + FOOT_FRACTION_LOSS_WEIGHT * foot_loss
                    + SPACING_LOSS_WEIGHT * spacing_loss)

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
                foot_fraction = batch["foot_fraction"].to(DEVICE)
                avg_spacing = batch["avg_spacing"].to(DEVICE)

                logits = model(input_ids, grade, angle, labels=labels)
                ce_loss = criterion(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1))
                length_loss = nn.functional.mse_loss(model.predict_length(grade, angle), num_holds)
                foot_loss = nn.functional.mse_loss(model.predict_foot_fraction(grade, angle), foot_fraction)
                spacing_loss = nn.functional.mse_loss(model.predict_spacing(grade, angle), avg_spacing)
                val_loss += (ce_loss + LENGTH_LOSS_WEIGHT * length_loss + FOOT_FRACTION_LOSS_WEIGHT * foot_loss
                             + SPACING_LOSS_WEIGHT * spacing_loss).item()

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

    print("\nEvaluating best checkpoint (generation-quality metrics, not just loss)...")
    model.load_state_dict(load_checkpoint_state_dict("kilter_setter_best.pt", map_location=DEVICE))
    run_evaluation(model, device=DEVICE, label=f"training run, val_loss={best_val_loss:.4f}",
                   checkpoint_path="kilter_setter_best.pt", val_loss=best_val_loss)


if __name__ == "__main__":
    train()
