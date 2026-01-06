import torch
from torchvision import transforms, models
import cv2
from PIL import Image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load trained model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("pothole_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

class_names = ["no_pothole", "pothole"]

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL image for model
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        label = class_names[predicted.item()]

    # Draw prediction text
    color = (0, 0, 255) if label == "pothole" else (0, 255, 0)
    cv2.putText(frame, f"Prediction: {label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show video stream
    cv2.imshow("Live Pothole Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
