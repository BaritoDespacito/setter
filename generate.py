import torch
from PIL import Image, ImageDraw
from preprocessing import parseRow, TYPE_TO_TOKEN, START_TOKEN_ID, END_TOKEN_ID, PAD_TOKEN_ID
from setter import Setter
import argparse
from torch.utils.data import Dataset

TYPE_TO_COLOR = {
    12: "lime",
    13: "cyan",
    14: "fuchsia",
    15: "orange",
}

def generate_route(model, grade, angle, max_length=50, temperature=0.8, device="cpu"):
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

        probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # Random sample
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


def drawClimb(holds):
    """Converts hold strings to a visual representation."""
    imgTemp = Image.open("kilterboardImg.jpg")
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
            # print("hold_id:", hold_id, "hold_type:", hold_type)
            if 1090 <= hold_id <= 1395: # bolt ons
                hold_id -= 1090
                x, y = (hold_id % 17) * 71 + 75, 1413 - ((hold_id // 17) * 71) - 135
                # print("hold_id:", hold_id, "x:", x, "y:", y)
                # print(hold_id % 17, hold_id // 17)
                draw.circle((x, y), 35, outline=TYPE_TO_COLOR[hold_type], width=7)
            elif 1448 <= hold_id <= 1465: # kickboard footholds
                hold_id -= 1448
                x, y = (hold_id % 18) * 71 + 36, 1385
                draw.circle((x, y), 25, outline=TYPE_TO_COLOR[hold_type], width=6)
            elif 1466 <= hold_id <= 1600: # footholds
                hold_id -= 1466
                x, y = (hold_id % 9) * 142 + 35 + ((hold_id // 9) % 2) * 71, 1413 - ((hold_id // 9) * 71) - 170
                draw.circle((x, y), 25, outline=TYPE_TO_COLOR[hold_type], width=6)
            elif 1073 <= hold_id <= 1089: # kickboard bolt ons
                hold_id -= 1073
                x, y = 1284 - ((hold_id % 17) * 71 + 75), 1349
                # print("hold_id:", hold_id, "x:", x, "y:", y)
                print(hold_id % 17, hold_id // 17)
                draw.circle((x, y), 35, outline=TYPE_TO_COLOR[hold_type], width=7)

    img.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("grade", type=int, help="Grade of the route (V1-V16)")
    parser.add_argument("angle", type=int, help="Angle of the route (0-70 degrees)")
    args = parser.parse_args()

    print(f"Generating climb... (Grade: {args.grade}, Angle: {args.angle})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Using device:', device)
    # device = "cpu"
    model = Setter(vocab_size=16000).to(device)
    model.load_state_dict(torch.load("kilter_setter_epoch_9.pt"))

    tokens = generate_route(model, grade=args.grade, angle=args.angle, device=device)
    climb = decode_holds(tokens)

    print("Generated climb:", " → ".join(climb))
    drawClimb(climb)