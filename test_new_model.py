"""
Quick test to verify the new model architecture works
"""
import torch
from setter import Setter

def test_model():
    print("Testing new Setter model architecture...")

    # Create model
    model = Setter(vocab_size=16000, d_model=256, nhead=8, num_layers=6)
    print(f"✓ Model created successfully")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass with training data
    batch_size = 4
    seq_len = 10
    label_len = 15

    input_ids = torch.randint(0, 16000, (batch_size, seq_len))
    labels = torch.randint(0, 16000, (batch_size, label_len))
    grade = torch.rand(batch_size)
    angle = torch.rand(batch_size)

    print(f"\nTesting training mode (teacher forcing)...")
    model.train()
    logits = model(input_ids, grade, angle, labels=labels)
    print(f"✓ Training forward pass successful")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Output logits shape: {logits.shape}")

    # Test inference mode
    print(f"\nTesting inference mode...")
    model.eval()
    with torch.no_grad():
        # Start with just START token
        input_ids_inf = torch.tensor([[4]])  # START_TOKEN_ID
        grade_inf = torch.tensor([0.5])
        angle_inf = torch.tensor([0.5])

        logits_inf = model(input_ids_inf, grade_inf, angle_inf, labels=None)
        print(f"✓ Inference forward pass successful")
        print(f"  Input shape: {input_ids_inf.shape}")
        print(f"  Output logits shape: {logits_inf.shape}")

        # Test autoregressive generation (2 steps)
        next_token = torch.argmax(logits_inf[:, -1], dim=-1, keepdim=True)
        input_ids_inf = torch.cat([input_ids_inf, next_token], dim=1)
        logits_inf = model(input_ids_inf, grade_inf, angle_inf, labels=None)
        print(f"✓ Autoregressive step successful")
        print(f"  New input shape: {input_ids_inf.shape}")
        print(f"  New output logits shape: {logits_inf.shape}")

    print(f"\n✅ All tests passed! Model is ready for training.")
    return True

if __name__ == "__main__":
    try:
        test_model()
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
