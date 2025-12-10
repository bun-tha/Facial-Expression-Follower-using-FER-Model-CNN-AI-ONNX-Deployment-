import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np

# Configuration
model_path = 'final_b0.pth'
camera_index = 0
num_classes = 5
class_names = ['angry', 'happy', 'neutral', 'sad', 'surprised']

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load model
model = models.efficientnet_b0(pretrained=False)

# Load the state dict first to inspect it
state_dict = torch.load(model_path, map_location=device)

# Check the keys to understand the structure
print("Checking classifier keys in saved model:")
classifier_keys = [k for k in state_dict.keys() if 'classifier' in k]
for key in classifier_keys:
    print(f"  {key}")

# Based on the keys, the saved model has classifier.1.1.weight and classifier.1.1.bias
# This means the Sequential has: [0] Dropout, [1] Sequential containing Linear
# So we need to match that exact structure
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_features, num_classes)
)

# Now load the state dict
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print('✓ Model loaded successfully')

# Transforms (same as validation)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open camera
cap = cv2.VideoCapture(camera_index)
print('Starting camera... Press "q" to quit')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    # Process each face
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        # Preprocess
        face_tensor = transform(face_roi).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(face_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            emotion = class_names[predicted.item()]
            conf = confidence.item() * 100
        
        # Draw rectangle and text
        color = (0, 255, 0) if conf > 60 else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Display emotion and confidence
        text = f'{emotion}: {conf:.1f}%'
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Show frame
    cv2.imshow('Facial Expression Recognition', frame)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print('Camera closed')