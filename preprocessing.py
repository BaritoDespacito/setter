from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import re
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

class KilterDataset(Dataset):
    def __init__(self, data):
        self.data = [parseRow(x) for x in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

dataset = load_dataset("ilsenatorov/kilterboard")
print(dataset['train'][0])

# SPECIAL TOKENS
START_TOKEN = "<START>"
END_TOKEN = "<END>"
PAD_TOKEN = "<PAD>"

# HOLD TYPE TOKENS
"""
12, green, start
13, cyan, handholds
14, purple, finish
15, orange, footholds
"""
TYPE_TO_TOKEN = {
    12: 0,
    13: 1,
    14: 2,
    15: 3,
}

tokenizer = AutoTokenizer.from_pretrained("t5-small")
tokenizer.add_tokens([START_TOKEN, END_TOKEN, PAD_TOKEN])

train_dataset = KilterDataset(dataset["train"])
val_dataset = KilterDataset(dataset["validation"])

def parseRow(row):
    """
    Parses hold sequence from dataset into tokens.
    :param row: a list representing a row in the dataset
    :return: a dict, containing the sequence of holds, angle and grade
    """
    holds = re.findall(r'p(\d+)r(\d+)', row['text'])
    sequence = []
    for hold in holds:
        sequence.append((int(hold[0]), TYPE_TO_TOKEN[int(hold[1])]))
    sequence = [START_TOKEN] + sequence + [END_TOKEN]
    angle = float(row['angle'])/70.0
    grade = (float(row['difficulty'])-10.0)/21.0

    return {
        'sequence': sequence,
        'angle': torch.tensor(angle),
        'grade': torch.tensor(grade),
    }

def collate_fn(batch):
    """Pad sequences and combine grade/angle inputs"""
    padded_batch = {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x["input_ids"]) for x in batch],
            padding_value=tokenizer.pad_token_id,
            batch_first=True
        ),
        "labels": torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x["labels"]) for x in batch],
            padding_value=tokenizer.pad_token_id,
            batch_first=True
        ),
        "grade": torch.stack([x["grade"] for x in batch]),
        "angle": torch.stack([x["angle"] for x in batch])
    }
    return padded_batch

sample = train_dataset[0]
print("Input IDs:", sample["input_ids"])
print("Labels:", sample["labels"])
print("Grade (normalized):", sample["grade"])
print("Angle (normalized):", sample["angle"])