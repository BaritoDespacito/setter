"""
Trains the RouteCritic: given a full real route + angle, predict its difficulty.
Reuses preprocessing.py's existing train_dataset/test_dataset directly - no new data
pipeline needed, since concatenating each example's input_ids + labels always
reconstructs the complete original route regardless of whether that example went
through the truncation augmentation.
"""
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from preprocessing import train_dataset, test_dataset
from setter import PAD_TOKEN_ID, normalized_difficulty_to_v_grade
from critic import RouteCritic
from tqdm import tqdm

BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
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


class CriticDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        route = torch.cat([item["input_ids"], item["labels"]])
        return {"route": route, "angle": item["angle"], "grade": item["grade"]}


def collate_fn(batch):
    return {
        "route": torch.nn.utils.rnn.pad_sequence(
            [x["route"] for x in batch], padding_value=PAD_TOKEN_ID, batch_first=True
        ),
        "angle": torch.stack([x["angle"] for x in batch]),
        "grade": torch.stack([x["grade"] for x in batch]),
    }


def train():
    set_seed(SEED)

    train_loader = DataLoader(
        CriticDataset(train_dataset), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        CriticDataset(test_dataset), batch_size=BATCH_SIZE, collate_fn=collate_fn
    )

    model = RouteCritic().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            route = batch["route"].to(DEVICE)
            angle = batch["angle"].to(DEVICE)
            grade = batch["grade"].to(DEVICE)

            optimizer.zero_grad()
            pred = model(route, angle)
            loss = nn.functional.mse_loss(pred, grade)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        abs_diff_sum = 0.0
        v_grade_correct = 0
        v_grade_within_1 = 0
        n = 0
        with torch.no_grad():
            for batch in test_loader:
                route = batch["route"].to(DEVICE)
                angle = batch["angle"].to(DEVICE)
                grade = batch["grade"].to(DEVICE)

                pred = model(route, angle)
                val_loss += nn.functional.mse_loss(pred, grade).item()

                for p, g in zip(pred.tolist(), grade.tolist()):
                    pred_v = normalized_difficulty_to_v_grade(p)
                    true_v = normalized_difficulty_to_v_grade(g)
                    abs_diff_sum += abs(pred_v - true_v)
                    v_grade_correct += pred_v == true_v
                    v_grade_within_1 += abs(pred_v - true_v) <= 1
                    n += 1

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        mae_v_grade = abs_diff_sum / n
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"V-grade MAE: {mae_v_grade:.2f} | exact: {v_grade_correct/n*100:.0f}% | within 1: {v_grade_within_1/n*100:.0f}%")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "critic_best.pt")
            print(f"  New best val loss ({best_val_loss:.4f}) -> saved critic_best.pt")


if __name__ == "__main__":
    train()
