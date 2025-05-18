import torch
from PIL import Image, ImageDraw
from preprocessing import parseRow, TYPE_TO_TOKEN, START_TOKEN_ID, END_TOKEN_ID, PAD_TOKEN_ID
from setter import Setter
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
        else:
            hold_id = int(hold[1:hold.index("r")])
            hold_type = int(hold[hold.index("r") + 1:])
            # print("hold_id:", hold_id, "hold_type:", hold_type)
            if 1090 <= hold_id <= 1395:
                hold_id -= 1090
                x, y = (hold_id % 17) * 71 + 75, 1413 - ((hold_id // 17) * 71) - 135
                print("hold_id:", hold_id, "x:", x, "y:", y)
                print(hold_id % 17, hold_id // 17)
                draw.circle((x, y), 35, outline=TYPE_TO_COLOR[hold_type], width=7)
            else:
                x, y = 1, 1


    # draw.circle((100, 100), 30, fill="green", outline="black", width=10)

    img.show()

if __name__ == "__main__":
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    # model = Setter(vocab_size=16000).to(device)
    # model.load_state_dict(torch.load("kilter_setter_epoch_9.pt"))

    # tokens = generate_route(model, grade=5, angle=40, device=device)
    # climb = decode_holds(tokens)

    climb = ['[START]', 'p1100r14', 'p1101r14', 'p1116r14', 'p1117r14', 'p1111r15', 'p1114r12', 'p1147r12', 'p1200r13', 'p1236r13', 'p1286r13', 'p1340r13', 'p1391r14', 'p1455r15', 'p1457r15', 'p1462r15', 'p1477r15', 'p1516r15', 'p1513r15', 'p1553r15', 'p1518r15', 'p1521r15', 'p1533r15', 'p1508r15', 'p1533r15', 'p1530r15', 'p1546r14', 'p1523r13', 'p1536r15', 'p1504r15', 'p1525r15', 'p1532r15', 'p1534r15', 'p1547r15', 'p1534r15', 'p1541r13', 'p1541r13', 'p1561r15', 'p1215r13', 'p1529r15', 'p1129r15', 'p1512r12', 'p1561r13', 'p1506r15', 'p1552r13', 'p1559r13', 'p1535r15', 'p1242r13', 'p1556r15', 'p1541r15', 'p1517r15', 'p1539r15', 'p1533r15', 'p1522r15', 'p1541r13']
    print("Generated climb:", " → ".join(climb))
    drawClimb(climb)