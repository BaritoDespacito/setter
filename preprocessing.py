from datasets import load_dataset, DatasetDict
import torch
from torch.utils.data import Dataset, DataLoader
import re
import random
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

dataset = load_dataset("ilsenatorov/kilterboard", split="train")
dataset = dataset.train_test_split(test_size=0.2)
# print(dataset["train"][0])

# SPECIAL TOKENS
START_TOKEN = "<START>"
END_TOKEN = "<END>"
PAD_TOKEN = "<PAD>"
START_TOKEN_ID = 4
END_TOKEN_ID = 5
PAD_TOKEN_ID = 6

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
tokenizer.add_special_tokens({
    "bos_token":  START_TOKEN,
    "eos_token": END_TOKEN,
    "pad_token": PAD_TOKEN,
})
tokenizer.bos_token_id = START_TOKEN_ID
tokenizer.eos_token_id = END_TOKEN_ID
tokenizer.pad_token_id = PAD_TOKEN_ID

def parseRow(row, min_truncate=1):
    """
    Parses hold sequence from dataset into tokens.
    :param min_truncate: minimum holds to truncate the sequence at
    :param row: a list representing a row in the dataset
    :return: a dict, containing the input sequence of holds, the next hold, angle and grade
    """
    holds = re.findall(r'p(\d+)r(\d+)', row['text'])
    sequence = [START_TOKEN_ID]
    for hold in holds:
        hold_token = int(hold[0]) * 10 + TYPE_TO_TOKEN[int(hold[1])]
        sequence.append(hold_token)
    sequence.append(END_TOKEN_ID)

    truncate_at = random.randint(min_truncate, len(holds) - 1)
    input_ids = sequence[:truncate_at]
    # print("input_ids:", input_ids)
    labels = [sequence[truncate_at]]
    # print("labels:", labels)

    angle = float(row['angle'])/70.0
    grade = (float(row['difficulty'])-10.0)/21.0

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
        'angle': torch.tensor(angle),
        'grade': torch.tensor(grade),
    }

def collate_fn(batch):
    """
    Pads the input sequences and labels to the same length.
    :param batch: a list of dictionaries, each containing input_ids, labels, angle and grade
    :return: a dictionary with padded input_ids, labels, angle and grade
    """
    padded_batch = {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            [x["input_ids"] for x in batch],
            padding_value=tokenizer.pad_token_id,
            batch_first=True
        ),
        "labels": torch.nn.utils.rnn.pad_sequence(
            [x["labels"] for x in batch],
            padding_value=tokenizer.pad_token_id,
            batch_first=True
        ),
        "grade": torch.stack([x["grade"] for x in batch]),
        "angle": torch.stack([x["angle"] for x in batch])
    }
    return padded_batch

class KilterDataset(Dataset):
    def __init__(self, data):
        self.data = [parseRow(x) for x in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_dataset = KilterDataset(dataset["train"])
test_dataset = KilterDataset(dataset["test"])

sample = train_dataset[0]
# print("Input IDs:", sample["input_ids"])
# print("Labels:", sample["labels"])
# print("Grade (normalized):", sample["grade"])
# print("Angle (normalized):", sample["angle"])

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    collate_fn=collate_fn,
    shuffle=True
)

batch = next(iter(train_loader))
# print("Batch input_ids shape:", batch["input_ids"].shape)
# print("Batch labels shape:", batch["labels"].shape)
# print("Batch grades:", batch["grade"])
# print("Batch angles:", batch["angle"])
# print("Max ID in batch:", batch["input_ids"].max().item())
# print("Min ID in batch:", batch["input_ids"].min().item())