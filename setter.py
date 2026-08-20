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

# Normalization constant for the auxiliary spacing-prediction head (see
# Setter.predict_spacing and training.py). Real average consecutive-hold distance
# ranges from ~130px (easiest) to ~230px (hardest); dividing by this keeps the target
# in a similar numeric range to the other auxiliary targets.
SPACING_NORM = 200.0

# Number of coarse "skeleton" waypoints predicted by Setter.predict_waypoints and
# conditioned on during generation (see forward()'s waypoint_embed prefix) - a small,
# fixed-length spatial plan (real (x, y) board positions, evenly spaced by climbing
# order) the decoder can realize, instead of only ever nudging length/composition via
# scalar auxiliary heads that never re-enter the decoder. 5 is enough points to convey
# a route's rough vertical/horizontal shape without demanding the planner get every
# move right.
NUM_WAYPOINTS = 5

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


def normalized_difficulty_to_v_grade(normalized):
    """Inverts v_grade_to_normalized_difficulty: finds the nearest V-grade for a
    predicted normalized difficulty (e.g. the critic's output)."""
    difficulty = normalized * 21.0 + 10.0
    return min(V_GRADE_TO_DIFFICULTY, key=lambda v: abs(V_GRADE_TO_DIFFICULTY[v] - difficulty))

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
    # d_model=384/num_layers=8 (~31.4M params, up from 14.6M at 256/6) - a deliberate
    # scale-up now that the dataset has grown to ~256k routes and the model has to
    # juggle more signals (coordinate embedding, three auxiliary heads, climbing-order
    # sequences). NOT warm-startable from checkpoints at the old size - shapes differ.
    def __init__(self, vocab_size, d_model=384, nhead=8, num_layers=8):
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

        # Two-stage generation: waypoint_head predicts a coarse (x, y) skeleton from
        # grade/angle alone (see predict_waypoints), the same way length_head etc. do
        # below - but unlike those, this plan re-enters the decoder as real
        # conditioning (see forward()'s waypoint prefix) instead of only ever
        # configuring an external loop parameter. waypoint_embed projects the plan's
        # (x, y) positions the same way coord_embed does for placed-hold positions,
        # kept as a separate layer since the semantics differ (a plan position isn't a
        # placed hold). waypoint_pos_embed gives each of the NUM_WAYPOINTS prefix slots
        # its own identity, standing in for pos_encoding (which is step-in-sequence
        # positional, the wrong notion of "position" for a fixed small set of plan
        # slots that aren't sequence steps).
        self.waypoint_embed = nn.Linear(2, d_model)
        self.waypoint_pos_embed = nn.Parameter(torch.zeros(NUM_WAYPOINTS, d_model))

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
        # *distribution* (mean, log_std) of normalized hold count, not a single point
        # estimate. Real hold count is extremely variable even within one grade/angle
        # (a hard route can be short-and-powerful or long-and-sustained) - a plain MSE
        # target collapses to the conditional mean and pulls every generation toward
        # "medium length" regardless of how wide the real distribution actually is.
        # Predicting mean+std (trained via Gaussian NLL, see training.py) lets
        # generate.py sample a per-attempt target from the real spread instead.
        # Also still forces condition_embed to encode length-relevant information
        # (real routes get *shorter* as grade increases - see NUM_HOLDS_NORM) instead of
        # leaving the grade/angle signal free to be ignored in favor of easier-to-fit
        # token co-occurrence patterns.
        self.length_head = nn.Linear(d_model, 2)

        # Second auxiliary head, same idea: predicts the fraction of a route's holds
        # that are footholds, from the conditioning vector alone. Without this, nothing
        # constrains hold-*type* composition - the model could satisfy the length target
        # with any mix of hand/foot holds, and in practice was generating routes with
        # far more footholds than handholds (real data: footholds are always outnumbered
        # by handholds, foot/hand ratio 0.65-0.98 across grades).
        self.foot_fraction_head = nn.Linear(d_model, 1)

        # Third auxiliary head: predicts average consecutive-hold reach distance
        # (normalized) from the conditioning vector alone. Real data: this nearly
        # doubles from easiest to hardest grades (~130px to ~230px) - a difficulty
        # signal distinct from hold count, and one nothing else in the model
        # explicitly captures. Only meaningful now that preprocessing.py orders
        # holds by climbing height, so "consecutive" corresponds to a real move.
        self.spacing_head = nn.Linear(d_model, 1)

        # Predicts the NUM_WAYPOINTS-point coarse skeleton (see waypoint_embed above)
        # from the conditioning vector alone, trained via MSE against real extracted
        # waypoints (see training.py), consumed at inference time by generate.py to
        # condition stage-2 decoding. A bare nn.Linear here (the original design,
        # matching the other scalar aux heads) collapsed to predicting ~the dataset
        # mean regardless of grade/angle - confirmed post-first-retrain: MSE on held-
        # out data was statistically identical to an "always predict the mean"
        # baseline, and predicted x-variance was ~13x smaller than real data's. A
        # single scalar target (length, foot-fraction) apparently gets enough signal
        # through a bare Linear off `cond`, but a coherent 5-point 2D path evidently
        # doesn't - give it the same Linear->ReLU->Linear capacity condition_embed
        # itself uses instead of a linear readout of it.
        self.waypoint_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, NUM_WAYPOINTS * 2),
        )

        # Initialize weights
        self._init_weights()

    def predict_length(self, grade, angle):
        """Returns (mean, log_std) of the predicted normalized-hold-count
        distribution, each (batch,). log_std is clamped to keep the implied std
        within a sane range (~0.002 to ~1.0x NUM_HOLDS_NORM) - unclamped, Gaussian
        NLL loss can drive it toward -inf/+inf early in training when a row's
        residual happens to be tiny/huge, destabilizing the rest of the loss."""
        conditions = torch.stack([grade, angle], dim=-1)  # (batch, 2)
        cond = self.condition_embed(conditions)  # (batch, d_model)
        out = self.length_head(cond)  # (batch, 2)
        mean = out[..., 0]
        log_std = out[..., 1].clamp(-6.0, 0.0)
        return mean, log_std

    def predict_foot_fraction(self, grade, angle):
        conditions = torch.stack([grade, angle], dim=-1)  # (batch, 2)
        cond = self.condition_embed(conditions)  # (batch, d_model)
        return self.foot_fraction_head(cond).squeeze(-1)  # (batch,)

    def predict_spacing(self, grade, angle):
        conditions = torch.stack([grade, angle], dim=-1)  # (batch, 2)
        cond = self.condition_embed(conditions)  # (batch, d_model)
        return self.spacing_head(cond).squeeze(-1)  # (batch,)

    def predict_waypoints(self, grade, angle):
        conditions = torch.stack([grade, angle], dim=-1)  # (batch, 2)
        cond = self.condition_embed(conditions)  # (batch, d_model)
        return self.waypoint_head(cond).view(-1, NUM_WAYPOINTS, 2)  # (batch, NUM_WAYPOINTS, 2)

    def _init_weights(self):
        """Initialize weights for better training"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, grade, angle, waypoints, labels=None):
        """
        Args:
            input_ids: (batch, seq_len) - input sequence tokens
            grade: (batch,) - normalized grade values
            angle: (batch,) - normalized angle values
            waypoints: (batch, NUM_WAYPOINTS, 2) - normalized (x, y) coarse-skeleton
                positions this generation should realize (real extracted waypoints
                during training/teacher-forcing, predict_waypoints() + inference-time
                noise during generation - see generate.py). Prepended as extra decoder
                positions, not passed through fc_out (see step 6 below).
            labels: (batch, label_len) - target sequence tokens for training

        Returns:
            logits: (batch, seq_len, vocab_size) - token-position logits only, the
                same shape/meaning as before waypoints existed (the prepended
                waypoint positions are sliced off before returning, so every existing
                caller stays unaware anything was prepended).
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

        # 3b. Embed the waypoint plan as NUM_WAYPOINTS extra positions prepended in
        # front of the token sequence - self-attention lets every token position
        # attend back to the full plan (see the unmasked block in step 4), the same
        # way tokens already attend back to [START]. waypoint_pos_embed stands in for
        # pos_encoding here (these aren't sequence-step positions); cond_emb is added
        # for the same reason it's broadcast into every token position above.
        wp_emb = self.waypoint_embed(waypoints)  # (batch, NUM_WAYPOINTS, d_model)
        wp_emb = wp_emb + self.waypoint_pos_embed.unsqueeze(0) + cond_emb
        full_emb = torch.cat([wp_emb, tgt_emb], dim=1)  # (batch, NUM_WAYPOINTS+tgt_len, d_model)

        # 4. Create causal mask to prevent attending to future tokens. Token
        # positions can always see the full waypoint block (a plan revealed all at
        # once, not a sequence with an order of its own) and keep the standard
        # causal structure relative to each other. Waypoints can see each other but
        # NOT any token position - a plan should be a fixed thing tokens read from,
        # not something that reads back from what's been generated so far. This
        # isn't just semantics: it's what makes the waypoint block's key/value
        # projections actually invariant to future tokens, which is what
        # forward_cache_init/forward_cache_step's KV-cache relies on - if waypoints
        # could see tokens, their cached representation would go stale every time a
        # new token was generated, silently breaking the cache.
        tgt_len = tgt_emb.size(1)
        full_len = NUM_WAYPOINTS + tgt_len
        causal_mask = torch.zeros(full_len, full_len, device=tgt_emb.device)
        causal_mask[NUM_WAYPOINTS:, NUM_WAYPOINTS:] = nn.Transformer.generate_square_subsequent_mask(
            tgt_len,
            device=tgt_emb.device
        )
        causal_mask[:NUM_WAYPOINTS, NUM_WAYPOINTS:] = float("-inf")

        # 5. Create padding mask for target sequence - waypoints are never padding.
        tgt_padding_mask = (tgt_ids == PAD_TOKEN_ID)
        wp_padding_mask = torch.zeros(batch_size, NUM_WAYPOINTS, dtype=torch.bool, device=tgt_emb.device)
        full_padding_mask = torch.cat([wp_padding_mask, tgt_padding_mask], dim=1)

        # 6. Apply transformer decoder
        # tgt = waypoint plan + target sequence, memory = conditioning (grade/angle)
        # Cross-attention allows model to attend to conditioning at each step
        out = self.transformer_decoder(
            tgt=full_emb,
            memory=cond_emb,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=full_padding_mask
        )

        # 7. Project to vocabulary, dropping the waypoint positions - callers only
        # ever want token-position logits, exactly as before waypoints existed.
        logits = self.fc_out(out)[:, NUM_WAYPOINTS:]

        return logits

    # ---- KV-cached incremental decoding -----------------------------------------
    #
    # forward() above recomputes attention over the *entire* sequence built so far at
    # every decode step - correct, but wasteful: a causal transformer's attention
    # output at position i never changes once computed, so redoing it at every
    # subsequent step turns an O(n) generation into O(n^2). The methods below give
    # generate.py an incremental path: forward_cache_init() processes the waypoint
    # prefix + START once, forward_cache_step() processes exactly one new token per
    # call against a cache of every previous position's key/value projections, so
    # each step only does O(1) new attention work instead of O(n).
    #
    # nn.TransformerDecoder/TransformerDecoderLayer don't expose a cache-aware
    # forward in stable PyTorch, so this reuses each layer's own learned weights
    # (self_attn.in_proj_weight/out_proj, multihead_attn, linear1/2, norm1/2/3) via
    # manual tensor ops replicating the same post-norm recipe forward() relies on
    # (self-attn -> add&norm -> cross-attn -> add&norm -> FFN -> add&norm), rather
    # than reimplementing the model. No explicit causal mask is needed anywhere in
    # this path: the cache only ever holds *past* positions by construction (each
    # step appends exactly the position it just computed), and the waypoint prefix
    # is seeded into the cache once at init and never attends to tokens (see
    # forward()'s mask comment - this is precisely what keeps it valid to cache) -
    # so "can only attend to past + self" falls out of the incremental structure
    # itself instead of needing an explicit mask, unlike the full-sequence path above.
    #
    # Verified against forward() for numerical equivalence in generate.py's smoke
    # tests - see the "KV-cache equivalence" check there before trusting this path.

    def _embed_prefix(self, grade, angle, waypoints):
        """Builds the conditioning vector and waypoint-prefix embeddings - identical
        computation to steps 1 and 3b of forward(), factored out since both the
        cached and non-cached paths need it. Returns (cond_emb (batch,1,d_model),
        wp_emb (batch, NUM_WAYPOINTS, d_model))."""
        conditions = torch.stack([grade, angle], dim=-1)
        cond_emb = self.condition_embed(conditions).unsqueeze(1)
        wp_emb = self.waypoint_embed(waypoints)
        wp_emb = wp_emb + self.waypoint_pos_embed.unsqueeze(0) + cond_emb
        return cond_emb, wp_emb

    def _embed_token_step(self, token_ids, step_idx, cond_emb):
        """Embeds one new token-sequence position at absolute token-space index
        step_idx (0 for START, 1 for the next token, etc. - independent of the
        waypoint prefix's own separate positional embedding). Identical computation
        to forward()'s step 3, for a single position instead of the whole sequence."""
        tok_emb = self.embedding(token_ids) * math.sqrt(self.d_model)
        tok_emb = tok_emb + self.pos_encoding.pe[:, step_idx:step_idx + 1]
        tok_emb = tok_emb + cond_emb
        tok_emb = tok_emb + self.coord_embed(self.coord_table[token_ids])
        return tok_emb

    @staticmethod
    def _self_attn_cached(self_attn, x, cache_k, cache_v):
        """One layer's self-attention over x (batch, new_len, d_model) against
        cache_k/cache_v (batch, num_heads, past_len, head_dim), or None on the first
        call. Manually computes Q/K/V via self_attn's own in_proj_weight/bias (same
        combined-QKV parameter nn.MultiheadAttention itself uses for self-attention)
        so the learned weights are identical to the non-cached path - only the
        control flow (cache instead of full recompute) differs. Returns
        (attn_output, new_cache_k, new_cache_v) with the new position(s) appended."""
        num_heads = self_attn.num_heads
        d_model = x.size(-1)
        head_dim = d_model // num_heads
        batch, new_len, _ = x.shape

        w_q, w_k, w_v = self_attn.in_proj_weight.chunk(3, dim=0)
        b_q, b_k, b_v = self_attn.in_proj_bias.chunk(3, dim=0)
        q = nn.functional.linear(x, w_q, b_q)
        k_new = nn.functional.linear(x, w_k, b_k)
        v_new = nn.functional.linear(x, w_v, b_v)

        def split_heads(t):
            return t.view(batch, -1, num_heads, head_dim).transpose(1, 2)

        q, k_new, v_new = split_heads(q), split_heads(k_new), split_heads(v_new)
        k = torch.cat([cache_k, k_new], dim=2) if cache_k is not None else k_new
        v = torch.cat([cache_v, v_new], dim=2) if cache_v is not None else v_new

        attn_scores = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ v  # (batch, num_heads, new_len, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(batch, new_len, d_model)
        attn_output = self_attn.out_proj(attn_output)
        return attn_output, k, v

    @classmethod
    def _decoder_layer_cached(cls, layer, x, memory, cache_k, cache_v):
        """One TransformerDecoderLayer's forward for the cached path: self-attn
        (cached) -> add&norm -> cross-attn to memory (always length 1, cheap enough
        to just recompute via the layer's own multihead_attn each step, no cache
        needed) -> add&norm -> FFN -> add&norm. Same post-norm recipe as the
        norm_first=False default forward() implicitly uses via nn.TransformerDecoder."""
        sa_out, new_k, new_v = cls._self_attn_cached(layer.self_attn, x, cache_k, cache_v)
        x = layer.norm1(x + layer.dropout1(sa_out))
        mha_out = layer.multihead_attn(x, memory, memory, need_weights=False)[0]
        x = layer.norm2(x + layer.dropout2(mha_out))
        ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
        x = layer.norm3(x + layer.dropout3(ff_out))
        return x, new_k, new_v

    def forward_cache_init(self, grade, angle, waypoints):
        """Seeds the cache with the waypoint prefix - self-attention among
        themselves only, since (per forward()'s mask) waypoints never attend to
        token positions - then processes START as the first token step via
        forward_cache_step itself, so "START sees all waypoints + itself" falls out
        of the normal incremental step logic rather than needing special-cased
        masking here. Returns (logits for the position after START (batch, 1,
        vocab_size), cache, cond_emb) - cache is a list of (k, v) per decoder layer;
        cond_emb is returned for reuse in forward_cache_step (cross-attention memory
        never changes mid-generation)."""
        cond_emb, wp_emb = self._embed_prefix(grade, angle, waypoints)

        cache = []
        x = wp_emb
        for layer in self.transformer_decoder.layers:
            x, k, v = self._decoder_layer_cached(layer, x, cond_emb, None, None)
            cache.append((k, v))

        start_ids = torch.full((grade.size(0), 1), START_TOKEN_ID, device=grade.device)
        logits, cache = self.forward_cache_step(start_ids, step_idx=0, cond_emb=cond_emb, cache=cache)
        return logits, cache, cond_emb

    def forward_cache_step(self, token_ids, step_idx, cond_emb, cache):
        """Processes exactly one new token (batch, 1) at absolute token-space
        position step_idx against an existing cache. Returns (logits (batch, 1,
        vocab_size), updated cache)."""
        x = self._embed_token_step(token_ids, step_idx, cond_emb)
        new_cache = []
        for layer, (k, v) in zip(self.transformer_decoder.layers, cache):
            x, new_k, new_v = self._decoder_layer_cached(layer, x, cond_emb, k, v)
            new_cache.append((new_k, new_v))
        logits = self.fc_out(x)
        return logits, new_cache