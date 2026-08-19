import torch
from PIL import Image, ImageDraw
from setter import (
    Setter,
    VOCAB_SIZE,
    NUM_HOLDS_NORM,
    TYPE_TO_TOKEN,
    TOKEN_TO_TYPE,
    START_TOKEN_ID,
    END_TOKEN_ID,
    PAD_TOKEN_ID,
    START_HOLD_TYPE,
    FINISH_HOLD_TYPE,
    FOOT_HOLD_TYPE,
    is_valid_hold_id,
    build_valid_token_mask,
    build_hold_adjacency,
    load_checkpoint_state_dict,
    hold_id_to_xy,
    v_grade_to_normalized_difficulty,
)
import argparse
import datetime
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TYPE_TO_COLOR = {
    12: "lime",
    13: "cyan",
    14: "fuchsia",
    15: "orange",
}

MIN_GRADE, MAX_GRADE = 1, 17
MIN_ANGLE, MAX_ANGLE = 0, 70

# Cached once per process; masks out every token that decode_holds() can't decode
# (invalid hold ids, unknown hold types, PAD, START) so generation can never sample
# something that would crash decoding downstream.
_VALID_TOKEN_MASK = build_valid_token_mask(VOCAB_SIZE)

# Cached once per process: hold_id -> set of hold_ids within reach of it, and a
# per-token lookup of which hold_id each vocab entry corresponds to (0 for non-hold
# tokens). Used to softly steer sampling toward holds connected to the route built so
# far - see _sample_tokens_batch.
_HOLD_ADJACENCY = build_hold_adjacency()
_HOLD_ID_OF_TOKEN = torch.arange(VOCAB_SIZE) // 10
_IS_FOOT_TOKEN = (torch.arange(VOCAB_SIZE) % 10) == TYPE_TO_TOKEN[FOOT_HOLD_TYPE]

# Logit penalty applied to holds not reachable from anything placed so far. Soft, not
# a hard mask: the base model was never trained with this constraint in mind, so a
# hard cutoff risks zeroing out every option the model actually favors on a given
# step. 4.0 (the first value tried) compounds across an entire generation's worth of
# steps and was strong enough to eliminate footholds from generated routes almost
# entirely. 1.0 was the next value tried and kept composition close to the unpenalized
# baseline, but a post-batching re-sweep against evaluate.py's default grid (once
# generating many candidates got cheap enough to make a real sweep practical) found
# 2.0 clearly better on both axes: valid_rate 92%->98%, critic grade-match error
# 3.86->3.43 raw difficulty units - strong enough spatial coherence pressure is doing
# more useful work here than the hard validity-filter rejection alone was.
GRAPH_ADJACENCY_PENALTY = 2.0

# Nucleus (top-p) sampling threshold: only sample from the smallest set of tokens
# whose cumulative probability reaches this, cutting off the long unlikely tail
# (typos-of-a-hold, wildly implausible jumps) without hard-capping to a fixed k.
# Lowered from 0.9 in the same post-batching sweep as GRAPH_ADJACENCY_PENALTY above -
# 0.85 trims a bit more of the unlikely tail, part of the same valid_rate/critic-error
# improvement (tested jointly with GRAPH_ADJACENCY_PENALTY=2.0, not independently).
TOP_P = 0.85

# Proportional bias toward/away from foot-only holds, nudging the running foot
# fraction toward the model's own predict_foot_fraction() target. Small and capped
# for the same reason GRAPH_ADJACENCY_PENALTY is small: logit adjustments compound
# across an entire generation, so even a modest per-step bias can have an outsized
# cumulative effect.
FOOT_BIAS_SCALE = 2.0
FOOT_BIAS_MAX = 1.0


def _nucleus_filter(probs, top_p):
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    drop = cumulative > top_p
    drop[..., 1:] = drop[..., :-1].clone()
    drop[..., 0] = False  # always keep at least the top token
    sorted_probs[drop] = 0.0
    filtered = torch.zeros_like(probs)
    filtered.scatter_(-1, sorted_idx, sorted_probs)
    return filtered / filtered.sum(dim=-1, keepdim=True)


def _sample_tokens_batch(model, grade, angle, max_length, temperatures, device, top_p=TOP_P):
    """
    Generates B independent attempts (B = len(temperatures)) at once, running every
    decode step as a single batched forward pass instead of B separate sequential
    generations - the actual cost of generation is the transformer forward pass, so
    batching the attempts (rather than looping over them) turns B sequential passes
    per step into 1, without changing what gets generated (same per-row sampling
    logic/state as the original single-sequence version, just carrying a batch
    dimension throughout).

    Rows that finish (emit END) before others keep getting fed PAD for the remaining
    steps so the whole batch stays a dense tensor; each row's own placed-hold
    bookkeeping simply stops advancing once that row is finished.
    """
    B = len(temperatures)
    grade_norm = v_grade_to_normalized_difficulty(grade)
    angle_norm = angle / 70.0

    input_ids = torch.full((B, 1), START_TOKEN_ID, device=device)
    grade_tensor = torch.full((B,), grade_norm, device=device)
    angle_tensor = torch.full((B,), angle_norm, device=device)
    temperature_tensor = torch.tensor(temperatures, device=device).unsqueeze(1)  # (B, 1)

    mask = _VALID_TOKEN_MASK.to(device)
    hold_id_of_token = _HOLD_ID_OF_TOKEN.to(device)
    is_foot_token = _IS_FOOT_TOKEN.to(device)

    with torch.no_grad():
        target_foot_fraction = model.predict_foot_fraction(grade_tensor, angle_tensor)
    target_foot_fraction = target_foot_fraction.clamp(0.0, 1.0)

    placed_hold_ids = [set() for _ in range(B)]
    placed_foot_count = [0] * B
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_length):
        if finished.all():
            break

        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                grade=grade_tensor,
                angle=angle_tensor,
                labels=None
            )

        next_logits = logits[:, -1].clone()
        next_logits[:, ~mask] = float("-inf")

        has_placed = torch.tensor([len(placed_hold_ids[b]) > 0 for b in range(B)], device=device)
        if has_placed.any():
            is_placed_batch = torch.zeros(B, next_logits.size(-1), dtype=torch.bool, device=device)
            reachable_batch = torch.zeros(B, next_logits.size(-1), dtype=torch.bool, device=device)
            for b in range(B):
                if finished[b] or not placed_hold_ids[b]:
                    continue
                placed_tensor = torch.tensor(list(placed_hold_ids[b]), device=device)
                # A physical hold can never legitimately appear twice (even with a
                # different role) - this is a hard rule, not a preference, so a hard
                # mask is correct here (unlike the graph-adjacency case below).
                is_placed_batch[b] = torch.isin(hold_id_of_token, placed_tensor)

                reachable = set()
                for hid in placed_hold_ids[b]:
                    reachable |= _HOLD_ADJACENCY.get(hid, set())
                if reachable:
                    reachable_tensor = torch.tensor(list(reachable), device=device)
                    reachable_batch[b] = torch.isin(hold_id_of_token, reachable_tensor)

            next_logits[is_placed_batch] = float("-inf")

            penalize = mask.unsqueeze(0).expand(B, -1).clone()
            penalize[:, END_TOKEN_ID] = False  # never penalize being able to stop
            penalize &= ~reachable_batch
            penalize[~has_placed] = False
            next_logits[penalize] -= GRAPH_ADJACENCY_PENALTY

            bias = torch.zeros(B, device=device)
            for b in range(B):
                if has_placed[b]:
                    current_fraction = placed_foot_count[b] / len(placed_hold_ids[b])
                    bias[b] = (target_foot_fraction[b] - current_fraction) * FOOT_BIAS_SCALE
            bias = bias.clamp(-FOOT_BIAS_MAX, FOOT_BIAS_MAX)
            next_logits[:, is_foot_token] += bias.unsqueeze(1)

        probs = torch.softmax(next_logits / temperature_tensor, dim=-1)
        probs = _nucleus_filter(probs, top_p)
        next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

        # Rows already finished keep emitting PAD regardless of what they'd have sampled,
        # so input_ids stays dense/correct-length without those rows influencing anything
        # further.
        next_token = torch.where(
            finished.unsqueeze(1), torch.full_like(next_token, PAD_TOKEN_ID), next_token
        )
        input_ids = torch.cat([input_ids, next_token], dim=1)

        for b in range(B):
            if finished[b]:
                continue
            token_id = next_token[b, 0].item()
            if token_id == END_TOKEN_ID:
                continue
            placed_hold_ids[b].add(token_id // 10)
            if token_id % 10 == TYPE_TO_TOKEN[FOOT_HOLD_TYPE]:
                placed_foot_count[b] += 1

        finished = finished | (next_token.squeeze(1) == END_TOKEN_ID)

    return [input_ids[b].tolist() for b in range(B)]


def _predict_target_length(model, grade, angle, device):
    grade_norm = v_grade_to_normalized_difficulty(grade)
    angle_norm = angle / 70.0
    with torch.no_grad():
        pred = model.predict_length(
            torch.tensor([grade_norm], device=device),
            torch.tensor([angle_norm], device=device),
        )
    return pred.item() * NUM_HOLDS_NORM


# Max pixel distance from a hold to its nearest neighbor in the route before it's
# treated as a "floating"/unreachable hold rather than part of a coherent line
# (grid spacing is 71-142px between adjacent holds). Loosened from 220 to 260 after
# checking real generated routes against this threshold: several routes were being
# marked invalid over isolated-hold distances of 224-267px, imperceptible outliers
# a few pixels past the old cutoff, while genuinely disconnected holds (500px+) still
# correctly fail at the new threshold too.
MAX_NEIGHBOR_DISTANCE = 260.0

# A route can legitimately have two start holds (one per hand) or two finish holds
# (matching to finish), but only if the pair sits close enough together to plausibly
# be reached by two hands at once - not scattered across the board. Loosened from 200
# to 280 for the same reason as MAX_NEIGHBOR_DISTANCE above: real two-hand starts were
# routinely landing at 225-293px and getting rejected despite looking fine.
TWO_HAND_MAX_SPAN = 280.0


def _distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _route_is_valid(holds):
    """
    A sensible climb has one or two start holds, one or two finish holds (a pair
    represents one hold per hand, and must sit within a plausible two-hand span of
    each other), no physical hold reused (even with a different role), and no hold
    sitting isolated far away from every other hold in the route.

    Deliberately does NOT constrain foothold/handhold composition: type-15 holds are
    "Foot Only" (the setter restricting a hold to feet), not an inventory of every
    place feet can go - hand holds can always double as feet, so a route with zero
    Foot Only holds is completely normal, and there's no meaningful ratio bound.
    """
    hold_entries = [h for h in holds if h not in ("[START]", "[END]")]
    if not hold_entries:
        return False

    seen_ids = set()
    start_positions = []
    finish_positions = []
    positions = []
    for hold in hold_entries:
        hold_id = int(hold[1:hold.index("r")])
        hold_type = int(hold[hold.index("r") + 1:])
        if hold_id in seen_ids:
            return False
        seen_ids.add(hold_id)
        xy = hold_id_to_xy(hold_id)
        if xy is None:
            return False
        if hold_type == START_HOLD_TYPE:
            start_positions.append(xy)
        elif hold_type == FINISH_HOLD_TYPE:
            finish_positions.append(xy)
        positions.append(xy)

    if not (1 <= len(start_positions) <= 2) or not (1 <= len(finish_positions) <= 2):
        return False
    if len(start_positions) == 2 and _distance(*start_positions) > TWO_HAND_MAX_SPAN:
        return False
    if len(finish_positions) == 2 and _distance(*finish_positions) > TWO_HAND_MAX_SPAN:
        return False
    for i, (x, y) in enumerate(positions):
        nearest = min(
            ((x - ox) ** 2 + (y - oy) ** 2) ** 0.5
            for j, (ox, oy) in enumerate(positions)
            if j != i
        )
        if nearest > MAX_NEIGHBOR_DISTANCE:
            return False

    return True


def _critic_grade_distances_batch(critic, token_lists, grade, angle, device):
    """How far off (in normalized difficulty) the critic thinks each candidate route
    is from the requested grade - lower is a better match. Batched: one forward pass
    over all candidates instead of one call per candidate. token_lists must all be the
    same length (true for anything coming out of _sample_tokens_batch, since every row
    in a batch shares the same seq_len)."""
    target_norm = v_grade_to_normalized_difficulty(grade)
    route_ids = torch.tensor(token_lists, device=device)
    angle_tensor = torch.full((len(token_lists),), angle / 70.0, device=device)
    with torch.no_grad():
        predicted_norm = critic(route_ids, angle_tensor)
    return (predicted_norm - target_norm).abs().tolist()


def generate_route(model, grade, angle, max_length=None, temperature=0.8, device="cpu", max_attempts=16, critic=None):
    """
    Generates a token-id route by sampling max_attempts candidates at once (one
    batched decode loop, not a sequential Python loop - see _sample_tokens_batch),
    each at a slightly higher temperature than the last, then picking the best one.

    Without a critic: among attempts that pass the validity filter, returns the one
    whose hold count is closest to the model's own predict_length target; if none
    pass, returns the closest-to-target attempt overall as a best-effort result,
    rather than just whichever attempt happened last.

    With a critic (see critic.py): the hold-count proxy is replaced with something more
    direct - the critic judges each *finished* route (also batched - one forward pass
    over every candidate) and whichever one it thinks actually looks closest to the
    requested grade wins. This is a better proxy for "does this route look right" than
    hold count, since the critic looks at the whole route bidirectionally rather than
    inferring from a single scalar target.

    If max_length isn't given, it's derived from the predict_length target plus slack,
    instead of a fixed cap - so a V1 and a V10 aren't generated with the same length
    budget.
    """
    if not (MIN_GRADE <= grade <= MAX_GRADE):
        raise ValueError(f"grade must be between {MIN_GRADE} and {MAX_GRADE}, got {grade}")
    if not (MIN_ANGLE <= angle <= MAX_ANGLE):
        raise ValueError(f"angle must be between {MIN_ANGLE} and {MAX_ANGLE}, got {angle}")

    model.eval()
    predicted_holds = _predict_target_length(model, grade, angle, device)
    if max_length is None:
        max_length = int(max(8, min(45, round(predicted_holds * 1.4) + 3)))

    temperatures = [temperature * (1.0 + 0.1 * attempt) for attempt in range(max_attempts)]
    all_tokens = _sample_tokens_batch(model, grade, angle, max_length, temperatures, device)

    candidates = []  # (tokens, valid, hold_count)
    for tokens in all_tokens:
        climb = decode_holds(tokens)
        valid = _route_is_valid(climb)
        hold_count = len([h for h in climb if h not in ("[START]", "[END]")])
        candidates.append((tokens, valid, hold_count))

    valid_candidates = [c for c in candidates if c[1]]
    pool = valid_candidates if valid_candidates else candidates

    if critic is not None:
        distances = _critic_grade_distances_batch(critic, [c[0] for c in pool], grade, angle, device)
        best = pool[min(range(len(pool)), key=lambda i: distances[i])]
    else:
        best = min(pool, key=lambda c: abs(c[2] - predicted_holds))
    return best[0]


def decode_holds(token_ids):
    """Converts token IDs back to hold strings (e.g., 'p1081r15')."""
    holds = []
    for token in token_ids:
        if token == START_TOKEN_ID:
            holds.append("[START]")
        elif token == END_TOKEN_ID:
            holds.append("[END]")
        elif token == PAD_TOKEN_ID:
            continue
        else:
            hold_id = token // 10
            hold_type = token % 10
            if hold_type not in TOKEN_TO_TYPE or not is_valid_hold_id(hold_id):
                # Shouldn't happen once sampling is masked, but guard decode anyway.
                continue
            holds.append(f"p{hold_id}r{TOKEN_TO_TYPE[hold_type]}")
    return holds


def drawClimb(holds, save=True):
    """Converts hold strings to a visual representation."""
    imgTemp = Image.open(os.path.join(SCRIPT_DIR, "kilterboardImg.jpg"))
    img = imgTemp.copy()
    draw = ImageDraw.Draw(img)

    for hold in holds:
        if hold == "[START]":
            continue
        elif hold == "[END]":
            break
        else:
            hold_id = int(hold[1:hold.index("r")])
            hold_type = int(hold[hold.index("r") + 1:])
            xy = hold_id_to_xy(hold_id)
            if xy is None:
                continue
            is_bolt_on = 1073 <= hold_id <= 1395  # bolt-ons + kickboard bolt-ons render larger
            radius, width = (35, 7) if is_bolt_on else (25, 6)
            draw.circle(xy, radius, outline=TYPE_TO_COLOR[hold_type], width=width)

    if save:
        out_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_climb.png"
        out_path = os.path.join(SCRIPT_DIR, out_name)
        img.show()
        img.save(out_path)
        print(out_path)

    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("grade", type=int, help=f"Grade of the route ({MIN_GRADE}-{MAX_GRADE})")
    parser.add_argument("angle", type=int, help=f"Angle of the route ({MIN_ANGLE}-{MAX_ANGLE} degrees)")
    args = parser.parse_args()

    print(f"Generating climb... (Grade: {args.grade}, Angle: {args.angle})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Using device:', device)
    model = Setter(vocab_size=VOCAB_SIZE).to(device)
    checkpoint_path = os.path.join(SCRIPT_DIR, "kilter_setter_best.pt")
    model.load_state_dict(load_checkpoint_state_dict(checkpoint_path, map_location=device))

    from critic import load_critic
    critic = load_critic(os.path.join(SCRIPT_DIR, "critic_best.pt"), device=device)
    print("Using critic-based reranking" if critic else "No critic checkpoint found, using hold-count proxy")

    tokens = generate_route(model, grade=args.grade, angle=args.angle, device=device, critic=critic)
    climb = decode_holds(tokens)

    print("Generated climb:", " → ".join(climb))
    drawClimb(climb)
