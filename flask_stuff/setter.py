import torch
import torch.nn as nn
import math

# SPECIAL TOKENS
START_TOKEN = "<START>"
END_TOKEN = "<END>"
PAD_TOKEN = "<PAD>"
START_TOKEN_ID = 4
END_TOKEN_ID = 5
PAD_TOKEN_ID = 6

# Hold tokens are encoded as hold_id * 10 + hold_type (hold_type in 0-3, see TYPE_TO_TOKEN).
# Valid Kilterboard hold ids run up to 1599 (see drawClimb's coordinate ranges), so the
# largest possible hold token is 1599 * 10 + 3 = 15993. VOCAB_SIZE must stay above that.
# (Kept at 16020, not tightened to match MAX_HOLD_ID exactly, so existing checkpoints'
# embedding/output layers - sized for the previous, slightly looser bound - still load.)
MAX_HOLD_ID = 1599
VOCAB_SIZE = 16020

# Normalization constant for the auxiliary length-prediction head (see Setter.predict_length
# and training.py). Real routes run up to ~20-25 holds; dividing by this keeps the target in
# a similar numeric range to the grade/angle inputs.
NUM_HOLDS_NORM = 20.0

# The model is trained on the dataset's raw "difficulty" field (~10-39, e.g. from
# preprocessing.py's `grade = (difficulty - 10) / 21.0`), NOT on the V-scale number
# climbers actually think in. generate.py's public API takes a V-grade (0-17+), so it
# must be converted to a representative raw difficulty before normalizing - passing a
# V-grade straight into the difficulty-based formula silently asks for the wrong
# thing (e.g. "V12" would normalize as difficulty=12, which is actually V0).
# Built from Kilter's own difficulty/V-grade table (Vilin97/KilterBoard's
# difficulty_grades table), averaged per V-grade and clipped to >=10 - the training
# data's actual minimum difficulty - so V0 doesn't map to an out-of-distribution value.
V_GRADE_TO_DIFFICULTY = {
    0: 11, 1: 14, 2: 15, 3: 16, 4: 18, 5: 20, 6: 22, 7: 23, 8: 24, 9: 26,
    10: 27, 11: 28, 12: 29, 13: 30, 14: 31, 15: 32, 16: 33, 17: 34,
}


def v_grade_to_normalized_difficulty(v_grade):
    """Converts a V-scale grade (as used by generate.py's public API) into the
    normalized difficulty value Setter's `grade` input actually expects."""
    difficulty = V_GRADE_TO_DIFFICULTY.get(
        v_grade, V_GRADE_TO_DIFFICULTY[max(V_GRADE_TO_DIFFICULTY)]
    )
    return (difficulty - 10) / 21.0

# HOLD TYPE TOKENS
# 12, green,  start
# 13, cyan,   handholds
# 14, purple, finish
# 15, orange, footholds
TYPE_TO_TOKEN = {
    12: 0,
    13: 1,
    14: 2,
    15: 3,
}
TOKEN_TO_TYPE = {v: k for k, v in TYPE_TO_TOKEN.items()}
START_HOLD_TYPE = 12
HAND_HOLD_TYPE = 13
FINISH_HOLD_TYPE = 14
FOOT_HOLD_TYPE = 15

# Valid Kilterboard hold-id ranges, as used by drawClimb()'s coordinate mapping.
# Verified against real usage frequency in the training dataset: bolt-on and
# kickboard-bolt-on boundaries check out exactly as below (clean cutoffs, no bleed
# into neighboring gaps). The foothold boundaries were off by one in an earlier
# version - hold 1447 was being treated as invalid/unrenderable despite being a
# very commonly used real foothold (usage frequency in line with its neighbors).
# The corrected split (1447-1464 / 1465-1599) is the only grid that divides the
# true foothold range (1447-1599, 153 ids) into a clean 18 + 15*9.
VALID_HOLD_ID_RANGES = [
    (1073, 1089),  # kickboard bolt-ons
    (1090, 1395),  # bolt-ons
    (1447, 1464),  # kickboard footholds
    (1465, 1599),  # footholds
]


def is_valid_hold_id(hold_id):
    return any(lo <= hold_id <= hi for lo, hi in VALID_HOLD_ID_RANGES)


def hold_id_to_xy(hold_id):
    """
    Maps a hold id to its pixel coordinates on kilterboardImg.jpg, using the same grid
    math as drawClimb(). Single source of truth so generate.py's route-coherence check
    and both drawClimb() implementations can't drift apart. Returns None if hold_id is
    outside the board.
    """
    if 1090 <= hold_id <= 1395:  # bolt-ons
        h = hold_id - 1090
        return (h % 17) * 71 + 75, 1413 - ((h // 17) * 71) - 135
    elif 1447 <= hold_id <= 1464:  # kickboard footholds
        h = hold_id - 1447
        return (h % 18) * 71 + 36, 1385
    elif 1465 <= hold_id <= 1599:  # footholds
        h = hold_id - 1465
        return (h % 9) * 142 + 35 + ((h // 9) % 2) * 71, 1413 - ((h // 9) * 71) - 170
    elif 1073 <= hold_id <= 1089:  # kickboard bolt-ons
        h = hold_id - 1073
        return 1284 - ((h % 17) * 71 + 75), 1349
    return None


# Max pixel distance between two holds for them to count as graph-adjacent (reachable
# from one another). Same threshold used for the route-coherence check in generate.py;
# grid spacing is 71-142px between adjacent holds, so this allows ~2 grid units of slack.
HOLD_ADJACENCY_DISTANCE = 220.0


def build_hold_adjacency(max_distance=HOLD_ADJACENCY_DISTANCE):
    """
    Maps each valid hold id to the set of other hold ids within max_distance of it.
    Used by generate.py to softly bias sampling toward holds connected to the route
    built so far, instead of relying purely on a post-hoc rejection filter.
    """
    hold_ids = [
        hold_id
        for lo, hi in VALID_HOLD_ID_RANGES
        for hold_id in range(lo, hi + 1)
    ]
    coords = {hid: hold_id_to_xy(hid) for hid in hold_ids}
    adjacency = {hid: set() for hid in hold_ids}
    for i, a in enumerate(hold_ids):
        ax, ay = coords[a]
        for b in hold_ids[i + 1:]:
            bx, by = coords[b]
            if ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= max_distance:
                adjacency[a].add(b)
                adjacency[b].add(a)
    return adjacency


# Normalization divisor for board pixel coordinates (see hold_id_to_xy), keeping the
# coordinate embedding's input in a similar numeric range to other model inputs.
BOARD_COORD_NORM = 1450.0


def build_coord_table(vocab_size=VOCAB_SIZE):
    """
    A (vocab_size, 2) table of normalized (x, y) board coordinates per token id, for
    Setter's coordinate embedding (see coord_embed in forward()). Special tokens
    (START/END/PAD) and any token that isn't a valid hold get (0, 0) - a neutral
    placeholder the model can distinguish from real positions via the token embedding
    it's added to.
    """
    table = torch.zeros(vocab_size, 2)
    for lo, hi in VALID_HOLD_ID_RANGES:
        for hold_id in range(lo, hi + 1):
            xy = hold_id_to_xy(hold_id)
            if xy is None:
                continue
            for type_token in TOKEN_TO_TYPE:
                token = hold_id * 10 + type_token
                if token < vocab_size:
                    table[token, 0] = xy[0] / BOARD_COORD_NORM
                    table[token, 1] = xy[1] / BOARD_COORD_NORM
    return table


def _migrate_vocab_size(state_dict, vocab_size):
    """
    Older checkpoints were trained with a smaller VOCAB_SIZE (16000, before the hold-id
    overflow fix bumped it to cover the full board). Rather than making those checkpoints
    unloadable, extend `embedding.weight`/`fc_out.weight`/`fc_out.bias` with freshly
    Xavier-initialized rows for the newly added token ids, preserving every weight the
    checkpoint already learned.
    """
    old_vocab_size = state_dict["embedding.weight"].shape[0]
    if old_vocab_size == vocab_size:
        return state_dict
    if old_vocab_size > vocab_size:
        raise ValueError(
            f"Checkpoint vocab_size ({old_vocab_size}) is larger than the current "
            f"VOCAB_SIZE ({vocab_size}); refusing to truncate learned embeddings."
        )

    print(
        f"Checkpoint was trained with vocab_size={old_vocab_size}; extending to "
        f"{vocab_size} with freshly-initialized rows for the new token ids."
    )
    state_dict = dict(state_dict)
    extra = vocab_size - old_vocab_size

    new_embedding = torch.empty(vocab_size, state_dict["embedding.weight"].shape[1])
    nn.init.xavier_uniform_(new_embedding)
    new_embedding[:old_vocab_size] = state_dict["embedding.weight"]
    state_dict["embedding.weight"] = new_embedding

    new_fc_weight = torch.empty(vocab_size, state_dict["fc_out.weight"].shape[1])
    nn.init.xavier_uniform_(new_fc_weight)
    new_fc_weight[:old_vocab_size] = state_dict["fc_out.weight"]
    state_dict["fc_out.weight"] = new_fc_weight

    new_fc_bias = torch.zeros(vocab_size)
    new_fc_bias[:old_vocab_size] = state_dict["fc_out.bias"]
    state_dict["fc_out.bias"] = new_fc_bias

    return state_dict


def load_checkpoint_state_dict(path, map_location="cpu", vocab_size=VOCAB_SIZE):
    """
    Loads a model state dict from a checkpoint file, transparently handling both the
    legacy format (torch.save(model.state_dict())) and the current format saved by
    training.py (a dict with a "model_state_dict" key alongside epoch/loss/optimizer info),
    and migrating older, smaller-vocab checkpoints so they still load cleanly.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    return _migrate_vocab_size(state_dict, vocab_size)


def build_valid_token_mask(vocab_size=VOCAB_SIZE):
    """
    Boolean mask over the vocabulary marking which token ids decode to something
    meaningful: END, or hold_id*10+type where hold_id is on the board and type is
    a known hold type. Used to keep sampling from ever producing a token that
    decode_holds() can't decode.
    """
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    mask[END_TOKEN_ID] = True
    for lo, hi in VALID_HOLD_ID_RANGES:
        for hold_id in range(lo, hi + 1):
            for type_token in TOKEN_TO_TYPE:
                token = hold_id * 10 + type_token
                if token < vocab_size:
                    mask[token] = True
    return mask

class PositionalEncoding(nn.Module):
    """Standard positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Setter(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        # Board-position embedding: gives the model an explicit geometric prior (each
        # hold's real x/y on the board) instead of having to infer board layout purely
        # from which hold-ids co-occur in training data. Coordinates are a fixed,
        # non-trainable buffer (see build_coord_table); only the projection into
        # d_model is learned.
        self.register_buffer("coord_table", build_coord_table(vocab_size))
        self.coord_embed = nn.Linear(2, d_model)

        # Conditioning: create rich embeddings for grade/angle
        self.condition_embed = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Use TransformerDecoder with cross-attention to conditioning
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

        # Auxiliary head, used only during training (see training.py): predicts the
        # normalized hold count of the full route from the conditioning vector alone.
        # This forces condition_embed to actually encode length-relevant information
        # (real routes get *shorter* as grade increases - see NUM_HOLDS_NORM) instead of
        # leaving the grade/angle signal free to be ignored in favor of easier-to-fit
        # token co-occurrence patterns.
        self.length_head = nn.Linear(d_model, 1)

        # Second auxiliary head, same idea: predicts the fraction of a route's holds
        # that are footholds, from the conditioning vector alone. Without this, nothing
        # constrains hold-*type* composition - the model could satisfy the length target
        # with any mix of hand/foot holds, and in practice was generating routes with
        # far more footholds than handholds (real data: footholds are always outnumbered
        # by handholds, foot/hand ratio 0.65-0.98 across grades).
        self.foot_fraction_head = nn.Linear(d_model, 1)

        # Initialize weights
        self._init_weights()

    def predict_length(self, grade, angle):
        conditions = torch.stack([grade, angle], dim=-1)  # (batch, 2)
        cond = self.condition_embed(conditions)  # (batch, d_model)
        return self.length_head(cond).squeeze(-1)  # (batch,)

    def predict_foot_fraction(self, grade, angle):
        conditions = torch.stack([grade, angle], dim=-1)  # (batch, 2)
        cond = self.condition_embed(conditions)  # (batch, d_model)
        return self.foot_fraction_head(cond).squeeze(-1)  # (batch,)

    def _init_weights(self):
        """Initialize weights for better training"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, grade, angle, labels=None):
        """
        Args:
            input_ids: (batch, seq_len) - input sequence tokens
            grade: (batch,) - normalized grade values
            angle: (batch,) - normalized angle values
            labels: (batch, label_len) - target sequence tokens for training

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size = input_ids.size(0)

        # 1. Create conditioning vector as "memory" for cross-attention
        conditions = torch.stack([grade, angle], dim=-1)  # (batch, 2)
        cond_emb = self.condition_embed(conditions).unsqueeze(1)  # (batch, 1, d_model)

        # 2. Determine target sequence
        if labels is not None:
            # Training mode: teacher forcing.
            # `input_ids` holds the real prompt/prefix for this example (just [START] for
            # full-route examples, or [START, hold1, ...] for the "continue a partial route"
            # augmentation in preprocessing.py). The token that should predict labels[0] is
            # the *last real (non-pad) token* of that prefix, not always START - input_ids is
            # right-padded, so we gather each row's true last token rather than assuming
            # position -1 is meaningful.
            non_pad = (input_ids != PAD_TOKEN_ID)
            last_real_idx = (non_pad.sum(dim=1) - 1).clamp(min=0)  # (batch,)
            last_prefix_token = input_ids.gather(1, last_real_idx.unsqueeze(1))  # (batch, 1)
            tgt_ids = torch.cat([last_prefix_token, labels[:, :-1]], dim=1)
        else:
            # Inference mode: use input_ids as target
            tgt_ids = input_ids

        # 3. Embed target sequence with positional encoding, then add the conditioning
        # vector at every position (on top of the cross-attention below) so grade/angle
        # have a direct, persistent influence on each step instead of only being
        # reachable through one cross-attention read against a single memory token.
        # Also add each token's real board position, so the model has an explicit
        # geometric prior instead of inferring layout purely from co-occurrence.
        tgt_emb = self.embedding(tgt_ids) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = tgt_emb + cond_emb
        tgt_emb = tgt_emb + self.coord_embed(self.coord_table[tgt_ids])

        # 4. Create causal mask to prevent attending to future tokens
        tgt_len = tgt_emb.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_len,
            device=tgt_emb.device
        )

        # 5. Create padding mask for target sequence
        tgt_padding_mask = (tgt_ids == PAD_TOKEN_ID)

        # 6. Apply transformer decoder
        # tgt = target sequence, memory = conditioning (grade/angle)
        # Cross-attention allows model to attend to conditioning at each step
        out = self.transformer_decoder(
            tgt=tgt_emb,
            memory=cond_emb,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        # 7. Project to vocabulary
        logits = self.fc_out(out)

        return logits