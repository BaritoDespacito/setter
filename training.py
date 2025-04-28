import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from preprocessing import train_dataset, test_dataset, collate_fn  # Import the datasets directly
from setter import Setter
from tqdm import tqdm

# Constants
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train():

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn
    )

    vocab_size = 16000 # max id = 1548?, multiply by 10 and add hold type, + special tokens (start, end, pad) + extra for leeway

    model = Setter(vocab_size=vocab_size).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(DEVICE)
            grade = batch["grade"].to(DEVICE)
            angle = batch["angle"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(input_ids, grade, angle, labels=labels)
            loss = criterion(logits.view(-1, vocab_size), labels.view(-1))

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                grade = batch["grade"].to(DEVICE)
                angle = batch["angle"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                logits = model(input_ids, grade, angle, labels=labels)
                val_loss += criterion(logits.view(-1, vocab_size), labels.view(-1)).item()

        print(
            f"Epoch {epoch + 1} | Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {val_loss / len(val_loader):.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), f"kilter_setter_epoch_{epoch}.pt")


if __name__ == "__main__":
    train()