"""
Convert the best FER (Facial Expression Recognition) model to ONNX format.

This script loads the trained EfficientNet-B0 model and exports it to ONNX format
for deployment in production environments.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# =====================================================
# Configuration
# =====================================================
MODEL_PATH = "final_b0.pth"     # <-- your file
ONNX_PATH = "fer_model.onnx"
NUM_CLASSES = 5
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OPSET_VERSION = 11


# =====================================================
# Model definition (match your training notebook)
# =====================================================
def create_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Create the same model architecture as used in training.
    IMPORTANT: we modify classifier[1] to be a Sequential(Dropout, Linear),
    not replace the whole classifier, to match the checkpoint keys like
    'classifier.1.1.weight'.
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)

    # Original EfficientNet-B0 classifier is:
    #   classifier = Sequential(
    #       (0): Dropout
    #       (1): Linear
    #   )
    #
    # In your training code you did:
    #   model.classifier[1] = nn.Sequential(Dropout, Linear)
    #
    # That makes:
    #   classifier = Sequential(
    #       (0): Dropout
    #       (1): Sequential(
    #               (0): Dropout
    #               (1): Linear
    #           )
    #   )
    #
    # So the checkpoint has keys like 'classifier.1.1.weight'.
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )

    return model


# =====================================================
# Conversion function
# =====================================================
def convert_to_onnx(
    model_path: str,
    onnx_path: str,
    img_size: int = IMG_SIZE,
    num_classes: int = NUM_CLASSES,
    device: str = DEVICE,
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from: {model_path}")

    # Create architecture that matches training
    model = create_model(num_classes=num_classes)

    # Load checkpoint (handle both styles)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print("Detected checkpoint with 'model_state_dict' key.")
        state_dict = checkpoint["model_state_dict"]
    else:
        print("Detected plain state_dict checkpoint.")
        state_dict = checkpoint

    # This should now match keys like 'classifier.1.1.weight'
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully on {device}")
    print(f"Architecture: EfficientNet-B0 (with nested classifier[1] = Sequential)")
    print(f"Num classes: {num_classes}")
    print(f"Input size: 3 x {img_size} x {img_size}")

    # Dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    print("\nExporting to ONNX...")
    print(f"Output path: {onnx_path}")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"✓ Model successfully exported to: {onnx_path}")

    # Optional verification
    print("\nVerifying ONNX model...")
    try:
        import onnx
        import onnxruntime

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")

        file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"✓ ONNX file size: {file_size_mb:.2f} MB")

        print("\nTesting ONNX Runtime inference...")
        ort_session = onnxruntime.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name

        test_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
        ort_outputs = ort_session.run(None, {input_name: test_input})
        print("✓ ONNX Runtime inference OK")
        print(f"  Output shape: {ort_outputs[0].shape}")

        print("\nComparing PyTorch vs ONNX outputs...")
        with torch.no_grad():
            torch_input = torch.from_numpy(test_input).to(device)
            torch_output = model(torch_input).cpu().numpy()

        max_diff = np.abs(torch_output - ort_outputs[0]).max()
        print(f"✓ Max difference: {max_diff:.6f}")
        if max_diff < 1e-4:
            print("✓ PyTorch and ONNX outputs match closely 🎯")
        else:
            print("⚠ Warning: Outputs differ more than expected")

    except ImportError:
        print("⚠ onnx or onnxruntime not installed.")
        print("  pip install onnx onnxruntime to enable verification.")

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)


def main():
    print("=" * 60)
    print("PyTorch → ONNX Conversion")
    print("=" * 60)
    print()

    try:
        convert_to_onnx(
            model_path=MODEL_PATH,
            onnx_path=ONNX_PATH,
            img_size=IMG_SIZE,
            num_classes=NUM_CLASSES,
            device=DEVICE,
        )
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        raise


if __name__ == "__main__":
    main()
