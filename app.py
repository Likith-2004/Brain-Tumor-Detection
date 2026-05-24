import os
import uuid
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify
from utils import generate_gradcam

# Configuration
UPLOAD_FOLDER = "static/uploads"
GRADCAM_FOLDER = "static/gradcam"
MODEL_PATH = "C:\\Users\\Likith\\Desktop\\Projects\\Brain_Tumor_App\\model\\resnet18_brain_tumor.pth"
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Create folders if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

# Initialize app
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file uploaded'})

    filename = str(uuid.uuid4()) + "_" + file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Preprocess image
    image = Image.open(filepath).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probs, dim=0)
        predicted_label = CLASS_NAMES[predicted_idx.item()]

    # Generate Grad-CAM
    gradcam_path = generate_gradcam(model, filepath, predicted_idx.item(), GRADCAM_FOLDER)
    gradcam_url = f"/static/gradcam/{os.path.basename(gradcam_path)}"

    return jsonify({
        "prediction": predicted_label,
        "confidence": f"{confidence.item() * 100:.2f}",
        "gradcam": gradcam_url
    })

if __name__ == '__main__':
    app.run(debug=True)