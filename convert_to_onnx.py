import torch
import onnxruntime as ort
from setter import Setter

START_TOKEN_ID = 4


def main():
    device = "cpu"  # Force CPU for ONNX export (avoid device mismatches)

    model = Setter(vocab_size=16000).to(device)
    model.load_state_dict(torch.load("kilter_setter_epoch_9.pt", map_location=device))
    model.eval()

    # Create dummy inputs ON THE SAME DEVICE as model
    dummy_input_ids = torch.ones((1, 5), dtype=torch.long, device=device) * START_TOKEN_ID
    dummy_grade = torch.tensor([0.5], dtype=torch.float32, device=device)  # Normalized grade
    dummy_angle = torch.tensor([0.5], dtype=torch.float32, device=device)  # Normalized angle

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_grade, dummy_angle),
        "setter_app/public/models/model.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids", "grade", "angle"],
        output_names=["logits"],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'seq_len'},
            'logits': {0: 'batch_size', 1: 'seq_len'}
        },
        verbose=True
    )

    # Validate the export
    ort_session = ort.InferenceSession("setter_app/public/models/model.onnx")
    print("ONNX model validated successfully!")


if __name__ == "__main__":
    main()