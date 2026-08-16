import torch
from PIL import Image, ImageDraw
from setter import (
    Setter,
    VOCAB_SIZE,
    TYPE_TO_TOKEN,
    TOKEN_TO_TYPE,
    START_TOKEN_ID,
    END_TOKEN_ID,
    PAD_TOKEN_ID,
    START_HOLD_TYPE,
    FINISH_HOLD_TYPE,
    is_valid_hold_id,
    build_valid_token_mask,
    load_checkpoint_state_dict,
    hold_id_to_xy,
)
import argparse
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


def _sample_tokens(model, grade, angle, max_length, temperature, device):
    grade_norm = (grade - 10) / 21.0
    angle_norm = angle / 70.0

    input_ids = torch.tensor([[START_TOKEN_ID]], device=device)
    grade_tensor = torch.tensor([grade_norm], device=device)
    angle_tensor = torch.tensor([angle_norm], device=device)

    mask = _VALID_TOKEN_MASK.to(device)

    for _ in range(max_length):
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                grade=grade_tensor,
                angle=angle_tensor,
                labels=None
            )

        next_logits = logits[:, -1].clone()
        next_logits[:, ~mask] = float("-inf")

        probs = torch.softmax(next_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # Random sample
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == END_TOKEN_ID:
            break

    return input_ids[0].tolist()


# Max pixel distance from a hold to its nearest neighbor in the route before it's
# treated as a "floating"/unreachable hold rather than part of a coherent line
# (grid spacing is 71-142px between adjacent holds; ~2 grid units of slack).
MAX_NEIGHBOR_DISTANCE = 220.0


def _route_is_valid(holds):
    """
    A sensible climb has exactly one start hold, exactly one finish hold, no physical
    hold reused (even with a different role), and no hold sitting isolated far away
    from every other hold in the route.
    """
    hold_entries = [h for h in holds if h not in ("[START]", "[END]")]
    if not hold_entries:
        return False

    seen_ids = set()
    start_count = 0
    finish_count = 0
    positions = []
    for hold in hold_entries:
        hold_id = int(hold[1:hold.index("r")])
        hold_type = int(hold[hold.index("r") + 1:])
        if hold_id in seen_ids:
            return False
        seen_ids.add(hold_id)
        if hold_type == START_HOLD_TYPE:
            start_count += 1
        elif hold_type == FINISH_HOLD_TYPE:
            finish_count += 1
        xy = hold_id_to_xy(hold_id)
        if xy is not None:
            positions.append(xy)

    if start_count != 1 or finish_count != 1:
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


def generate_route(model, grade, angle, max_length=50, temperature=0.8, device="cpu", max_attempts=8):
    """
    Generates a token-id route, retrying (with a higher temperature each attempt) until
    the decoded route has exactly one start hold, exactly one finish hold, and no
    duplicate holds - or max_attempts is exhausted, in which case the last attempt is
    returned as a best-effort result.
    """
    if not (MIN_GRADE <= grade <= MAX_GRADE):
        raise ValueError(f"grade must be between {MIN_GRADE} and {MAX_GRADE}, got {grade}")
    if not (MIN_ANGLE <= angle <= MAX_ANGLE):
        raise ValueError(f"angle must be between {MIN_ANGLE} and {MAX_ANGLE}, got {angle}")

    model.eval()
    tokens = None
    for attempt in range(max_attempts):
        attempt_temperature = temperature * (1.0 + 0.1 * attempt)
        tokens = _sample_tokens(model, grade, angle, max_length, attempt_temperature, device)
        if _route_is_valid(decode_holds(tokens)):
            return tokens
    return tokens


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


def drawClimb(holds):
    """Converts hold strings to a visual representation. Returns the PIL image (caller saves/serves it)."""
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

    tokens = generate_route(model, grade=args.grade, angle=args.angle, device=device)
    climb = decode_holds(tokens)

    print("Generated climb:", " → ".join(climb))
    img = drawClimb(climb)
    img.show()
