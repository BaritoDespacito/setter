from datasets import load_dataset, Dataset as HFDataset
from huggingface_hub import hf_hub_download
import torch
from torch.utils.data import Dataset, DataLoader
import re
import random
import sqlite3
import collections
from setter import (
    START_TOKEN,
    END_TOKEN,
    PAD_TOKEN,
    START_TOKEN_ID,
    END_TOKEN_ID,
    PAD_TOKEN_ID,
    TYPE_TO_TOKEN,
    NUM_HOLDS_NORM,
    SPACING_NORM,
    NUM_WAYPOINTS,
    BOARD_COORD_NORM,
    is_valid_hold_id,
    hold_id_to_xy,
)

# Pinned to specific dataset commits so training runs are reproducible even if the
# upstream HF datasets are later updated. Re-pin deliberately if you want newer data.
DATASET_REVISION = "acb78b6cdde080b7eab1182c51313696197a2938"
SECONDARY_DATASET_REVISION = "1178d1faecb8dc60bf5e80dcd6e71c553984f92c"
VILIN_DATASET_REVISION = "8a5a0f11675494c3615c2c96815af6bde93235f9"
DATASET_SPLIT_SEED = 42

_PROMPT_DIFFICULTY_ANGLE_RE = re.compile(r'difficulty (\d+) and angle (\d+)')
_ROUTE_STRING_RE = re.compile(r'^(?:p\d+r\d+)+$')


def _load_vilin_pairs():
    """
    Vilin97/KilterBoard is a direct export of Kilter's own database (layout_id=1, i.e.
    the exact board/role scheme this project targets), with 236k+ unique routes pooled
    across its train/val/test splits - by far the largest and richest of the three
    sources available. Its own split boundaries aren't used here: everything is pooled
    and re-split by _load_combined_dataset() below so all three sources share one
    consistent, reproducible split.
    """
    path = hf_hub_download(
        repo_id="Vilin97/KilterBoard",
        filename="kilter_splits.sqlite",
        repo_type="dataset",
        revision=VILIN_DATASET_REVISION,
    )
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    pairs = {}
    for table in ("kilter_train", "kilter_val", "kilter_test"):
        cur.execute(f"SELECT frames, angle, difficulty_numeric FROM {table} WHERE layout_id=1")
        for text, angle, difficulty in cur.fetchall():
            if text and _ROUTE_STRING_RE.match(text) and angle is not None and difficulty is not None:
                pairs[(text, float(angle))] = float(difficulty)
    conn.close()
    return pairs


def _load_combined_dataset():
    """
    Merges three sources into one deduplicated set, keyed by (route, angle) since
    difficulty is assigned per angle, not per route alone:
      - ilsenatorov/kilterboard (69,172 routes, last updated 2024-07)
      - gabriead/Kilterboard (177,858 routes, uploaded 2025-07, ~78k not in the above)
      - Vilin97/KilterBoard (236,263 routes pooled across its train/val/test splits,
        a direct Kilter database export - the largest and most authoritative source)
    Later sources overwrite earlier ones' difficulty on conflicts, since Vilin's
    crowd-averaged difficulty_numeric is the most precise value available. The three
    sources overlap heavily but each contributes real unique routes; using all three
    gets ~256k unique (route, angle) pairs vs ~236k from Vilin alone.
    """
    primary = load_dataset("ilsenatorov/kilterboard", split="train", revision=DATASET_REVISION)
    secondary = load_dataset("gabriead/Kilterboard", split="train", revision=SECONDARY_DATASET_REVISION)

    combined = {}
    for row in primary:
        combined[(row["text"], float(row["angle"]))] = row["difficulty"]
    for row in secondary:
        match = _PROMPT_DIFFICULTY_ANGLE_RE.search(row["input"])
        if not match:
            continue
        difficulty, angle = int(match.group(1)), int(match.group(2))
        combined[(row["target"], float(angle))] = float(difficulty)
    combined.update(_load_vilin_pairs())

    return HFDataset.from_list([
        {"text": text, "angle": angle, "difficulty": difficulty}
        for (text, angle), difficulty in combined.items()
    ])


dataset = _load_combined_dataset()
dataset = dataset.train_test_split(test_size=0.2, seed=DATASET_SPLIT_SEED)


def _evenly_spaced_indices(n, k):
    """k indices into a length-n sequence, evenly spaced by position (equivalent to
    round(np.linspace(0, n-1, k)) without adding a numpy dependency for this one use).
    Used to pick which holds' real positions become the coarse-skeleton waypoints -
    see parseRow. n<=1 repeats index 0 for every slot, matching linspace's behavior."""
    if n <= 1:
        return [0] * k
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]

def parseRow(row, min_truncate=1):
    """
    Parses hold sequence from dataset into tokens.
    :param min_truncate: minimum holds to truncate the sequence at (for data augmentation)
    :param row: a list representing a row in the dataset
    :return: a dict, containing the input sequence of holds, the full sequence labels, angle and grade
    """
    raw_holds = re.findall(r'p(\d+)r(\d+)', row['text'])
    valid_holds = []  # (hold_id_str, hold_type, hold_id_int, y)
    for hold_id, hold_type in raw_holds:
        hold_type = int(hold_type)
        hold_id_int = int(hold_id)
        if hold_type not in TYPE_TO_TOKEN:
            # Unrecognized hold-type role in the raw data; skip rather than crash the
            # whole dataset load on one malformed row.
            continue
        if not is_valid_hold_id(hold_id_int):
            # The merged dataset pools routes from multiple board configurations/sizes;
            # a small fraction (~0.5% of holds) reference hole ids far outside this
            # board's actual range (up to 4845, vs. our vocabulary's max of 1599) and
            # would otherwise overflow the embedding table. Skip just the bad hold
            # rather than dropping the whole route.
            continue
        _, y = hold_id_to_xy(hold_id_int)
        valid_holds.append((hold_id, hold_type, hold_id_int, y))

    # Order holds bottom-to-top (descending y - larger y is lower on the wall) as an
    # approximation of real climbing order, instead of the source data's arbitrary
    # listing order. This also makes the truncation augmentation below physically
    # meaningful (predict the rest of the climb from a real partial ascent) rather
    # than an arbitrary split, and gives the graph-adjacency decoding constraint in
    # generate.py a training-time sequence order it can actually line up with.
    valid_holds.sort(key=lambda h: -h[3])

    holds = []
    sequence = [START_TOKEN_ID]
    foot_count = 0
    for hold_id, hold_type, hold_id_int, _ in valid_holds:
        hold_token = hold_id_int * 10 + TYPE_TO_TOKEN[hold_type]
        sequence.append(hold_token)
        holds.append((hold_id, hold_type))
        if hold_type == 15:  # foot
            foot_count += 1
    sequence.append(END_TOKEN_ID)

    # Average distance between consecutive holds in (now climbing-order) sequence -
    # the real per-move reach - for the auxiliary spacing head in training.py. Real
    # data: this increases from ~130px at the easiest grades to ~230px at the hardest,
    # a signal nothing else in the model explicitly captures.
    if len(valid_holds) >= 2:
        coords = [hold_id_to_xy(h[2]) for h in valid_holds]
        dists = [
            ((coords[i][0] - coords[i + 1][0]) ** 2 + (coords[i][1] - coords[i + 1][1]) ** 2) ** 0.5
            for i in range(len(coords) - 1)
        ]
        avg_spacing = sum(dists) / len(dists)
    else:
        avg_spacing = 0.0

    # Coarse skeleton for two-stage generation (see Setter.predict_waypoints/forward
    # in setter.py): NUM_WAYPOINTS real (x, y) hold positions, evenly spaced by index
    # through the (already climbing-ordered) valid_holds list - a simple stand-in for
    # "evenly spaced by move count" that needs no extra design decisions about height
    # bins vs. move count, since holds are already in climbing order. Routes with
    # zero holds get an all-zero plan (matches coord_table's convention for "no real
    # position" elsewhere in setter.py); routes with fewer than NUM_WAYPOINTS holds
    # just repeat some indices, which is fine - it's still a real, if repeated, shape.
    if valid_holds:
        wp_indices = _evenly_spaced_indices(len(valid_holds), NUM_WAYPOINTS)
        waypoints = [hold_id_to_xy(valid_holds[i][2]) for i in wp_indices]
        waypoints = [(x / BOARD_COORD_NORM, y / BOARD_COORD_NORM) for x, y in waypoints]
    else:
        waypoints = [(0.0, 0.0)] * NUM_WAYPOINTS

    # Optional: Add data augmentation by sometimes truncating the input
    # This teaches the model to continue partial routes
    if random.random() < 0.3 and len(holds) > 3:  # 30% of time, truncate for augmentation
        truncate_at = random.randint(2, len(holds) - 1)
        input_ids = sequence[:truncate_at]
        labels = sequence[truncate_at:]  # Predict REST of sequence
    else:
        # Most of the time, generate full sequence from scratch
        input_ids = [START_TOKEN_ID]
        labels = sequence[1:]  # Predict entire route after START token

    angle = float(row['angle'])/70.0
    grade = (float(row['difficulty'])-10.0)/21.0

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
        'angle': torch.tensor(angle),
        'grade': torch.tensor(grade),
        # Full route hold count (independent of the truncation augmentation above),
        # normalized for the auxiliary length-prediction head in training.py.
        'num_holds': torch.tensor(len(holds) / NUM_HOLDS_NORM),
        # Fraction of the full route's holds that are footholds, for the auxiliary
        # foot-fraction head in training.py.
        'foot_fraction': torch.tensor(foot_count / len(holds) if holds else 0.0),
        # Average consecutive-hold distance in the (now climbing-order) sequence,
        # normalized, for the auxiliary spacing head in training.py.
        'avg_spacing': torch.tensor(avg_spacing / SPACING_NORM),
        # Coarse (x, y) skeleton, see above - shape (NUM_WAYPOINTS, 2).
        'waypoints': torch.tensor(waypoints, dtype=torch.float32),
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
            padding_value=PAD_TOKEN_ID,
            batch_first=True
        ),
        "labels": torch.nn.utils.rnn.pad_sequence(
            [x["labels"] for x in batch],
            padding_value=PAD_TOKEN_ID,
            batch_first=True
        ),
        "grade": torch.stack([x["grade"] for x in batch]),
        "angle": torch.stack([x["angle"] for x in batch]),
        "num_holds": torch.stack([x["num_holds"] for x in batch]),
        "foot_fraction": torch.stack([x["foot_fraction"] for x in batch]),
        "avg_spacing": torch.stack([x["avg_spacing"] for x in batch]),
        "waypoints": torch.stack([x["waypoints"] for x in batch]),
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

# Per-example sampling weights, inversely proportional to (the square root of, floored
# at MIN_BUCKET_SIZE) how common that example's (grade, angle) bucket is. Grade/angle
# are extremely imbalanced in the raw data (angle=40 alone is a large share of all
# routes, while rare grade/angle combos have only a handful of examples) - training
# with plain shuffling barely exposes the model to rare combinations.
#
# NOTE: a straight 1/count weighting was tried first and made things *worse* - with
# WeightedRandomSampler's replacement=True, singleton (count=1) buckets ended up with
# ~15x the expected draws per epoch of a "fair" pass, so the model just memorized that
# handful of examples through repetition instead of generalizing (val loss diverged
# from epoch 2 onward). sqrt() plus a count floor caps that at roughly 4x - enough to
# meaningfully rebalance exposure without turning rare examples into repeated near-labels.
#
# Bucket key rounds difficulty to the nearest integer: Vilin's difficulty_numeric is a
# continuous crowd-averaged value (e.g. 15.9958), unlike the other two sources' clean
# integer scale - without rounding, almost every Vilin example would be its own
# singleton bucket and the weighting would be meaningless. The actual `grade` value fed
# to the model (see parseRow) stays the precise unrounded float.
MIN_BUCKET_SIZE = 30
def _bucket_key(row):
    return (round(row["difficulty"]), row["angle"])

_bucket_counts = collections.Counter(_bucket_key(row) for row in dataset["train"])
train_sample_weights = [
    1.0 / (max(_bucket_counts[_bucket_key(row)], MIN_BUCKET_SIZE) ** 0.5)
    for row in dataset["train"]
]

if __name__ == "__main__":
    sample = train_dataset[0]
    print("Input IDs:", sample["input_ids"])
    print("Labels:", sample["labels"])
    print("Grade (normalized):", sample["grade"])
    print("Angle (normalized):", sample["angle"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        collate_fn=collate_fn,
        shuffle=True
    )
    batch = next(iter(train_loader))
    print("Batch input_ids shape:", batch["input_ids"].shape)
    print("Batch labels shape:", batch["labels"].shape)
    print("Batch grades:", batch["grade"])
    print("Batch angles:", batch["angle"])
    print("Max ID in batch:", batch["input_ids"].max().item())
    print("Min ID in batch:", batch["input_ids"].min().item())