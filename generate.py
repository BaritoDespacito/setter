import torch
from preprocessing import parseRow, TYPE_TO_TOKEN, START_TOKEN_ID, END_TOKEN_ID, PAD_TOKEN_ID
from setter import Setter
from torch.utils.data import Dataset


def generate_route(model, grade, angle, max_length=50, device="cpu"):
    model.eval()

    # Normalize inputs
    grade_norm = (grade - 10) / 21.0
    angle_norm = angle / 70.0

    # Initialize with [START]
    input_ids = torch.tensor([[START_TOKEN_ID]], device=device)
    grade_tensor = torch.tensor([grade_norm], device=device)
    angle_tensor = torch.tensor([angle_norm], device=device)

    for _ in range(max_length):
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                grade=grade_tensor,
                angle=angle_tensor,
                labels=None
            )

        next_token = logits.argmax(-1)[:, -1].unsqueeze(1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == END_TOKEN_ID:
            break

    return input_ids[0].tolist()


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

            type_map = {v: k for k, v in TYPE_TO_TOKEN.items()}
            holds.append(f"p{hold_id}r{type_map[hold_type]}")
    return holds


if __name__ == "__main__":
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model = Setter(vocab_size=16000).to(device)
    model.load_state_dict(torch.load("kilter_setter_epoch_9.pt"))

    tokens = generate_route(model, grade=5, angle=40, device=device)
    climb = decode_holds(tokens)
    print("Generated climb:", " → ".join(climb))