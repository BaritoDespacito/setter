from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import re

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

class KilterDataset(Dataset):
    def __init__(self, data):
        self.data = [parseRow(x) for x in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]