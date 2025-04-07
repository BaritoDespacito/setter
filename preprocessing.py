from datasets import load_dataset
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

def parseSequence(row):
    """
    Parses hold sequence from dataset into tokens.
    :param row: a string containing the sequence of holds
    :return: a list of tokens
    """
    holds = re.findall(r'p(\d+)r(\d+)', row['text'])
    sequence = []
    for hold in holds:
        sequence.append((int(hold[0]), TYPE_TO_TOKEN[int(hold[1])]))
    sequence = [START_TOKEN] + sequence + [END_TOKEN]
    angle = float(row['angle'])/70.0


parseSequence(dataset['train'][0])