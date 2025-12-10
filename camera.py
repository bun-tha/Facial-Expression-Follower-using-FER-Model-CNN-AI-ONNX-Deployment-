import cv2
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from PIL import Image
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ⚠️ Must match train_dataset.classes from your Colab
# (likely ['angry', 'happy', 'neutral', 'sad', 'surprised'])
class_names = ["angry", "happy", "neutral", "sad", "surprised"]

# -----------------------------
# 1. Define model and load weights
# -----------------------------
def load_model(weights_path="best_b0.pth"):
    """
    Load EfficientNet-B0 with the SAME structure as in your Colab notebook
    and load the checkpoint (with model_state_dict).
    """
    # Same base weights as in notebook: weights='IMAGENET1K_V1'
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)

    # Match this line from the notebook:
    # model.classifier[1] = nn.Sequential(Dropout(0.5), Linear(in_features, NUM_CLASSES))
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, len(class_names))
    )

    # Load checkpoint (you saved a dict with model_state_dict)
    checkpoint = torch.load(weights_path, map_location=DEVICE)

    # Handle both cases just in case:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # If you ever save plain state_dict
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded and ready on", DEVICE)
    return model

# -----------------------------
# 2. Preprocessing (match val_transform)
# -----------------------------
# From notebook:
# val_transform = Compose([
#   Resize((IMG_SIZE, IMG_SIZE)),
#   ToTensor(),
#   Normalize(mean, std)
# ])
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def preprocess_face(face_bgr):
    """BGR NumPy image -> preprocessed tensor [1, 3, H, W]"""
    # Convert BGR (OpenCV) to RGB
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)
    tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)  # [1, 3, 224, 224]
    return tensor

# -----------------------------
# 3. Real-time loop with OpenCV
# -----------------------------
def run_realtime(model):
    # Haar cascade from OpenCV install
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    if face_cascade.empty():
        print("⚠️ Could not load Haar cascade. Falling back to full-frame prediction.")
        use_face_detection = False
    else:
        use_face_detection = True

    # Try 0 first; change to 1 if your camera is on index 1
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    print("🎥 Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        # Mirror horizontally so it feels natural
        frame = cv2.flip(frame, 1)

        if use_face_detection:
            # Detect faces on grayscale image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(60, 60)
            )

            for (x, y, w, h) in faces:
                face_roi = frame[y:y + h, x:x + w]

                if face_roi.size == 0:
                    continue

                face_tensor = preprocess_face(face_roi)

                with torch.no_grad():
                    outputs = model(face_tensor)
                    probs = torch.softmax(outputs, dim=1)[0]
                    conf, pred_idx = torch.max(probs, dim=0)

                label = class_names[pred_idx.item()]
                confidence = conf.item() * 100.0

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Put text label
                text = f"{label} ({confidence:.1f}%)"
                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2, cv2.LINE_AA)
        else:
            # No face detection: run on full frame
            face_tensor = preprocess_face(frame)
            with torch.no_grad():
                outputs = model(face_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                conf, pred_idx = torch.max(probs, dim=0)

            label = class_names[pred_idx.item()]
            confidence = conf.item() * 100.0

            text = f"{label} ({confidence:.1f}%)"
            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow("Real-time Emotion Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Webcam closed")

# -----------------------------
# 4. Run everything
# -----------------------------
if __name__ == "__main__":
    # Use the same name you downloaded from Drive (likely best_b0.pth)
    model = load_model("best_b0.pth")  # change if you renamed the file
    run_realtime(model)
