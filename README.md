# 🧠 Brain Tumor Detection Application

<div align="center">

![Brain Tumor Detection](https://img.shields.io/badge/AI-Deep%20Learning-blue?style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-Model-red?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10-yellow?style=for-the-badge)

**An intelligent AI-powered web application for accurate brain tumor classification with explainable Grad-CAM visualizations**

[Features](#-features) • [Quick Start](#-quick-start) • [How It Works](#-how-it-works) • [Model Details](#model-details) • [Demo](#-demo)

</div>

---

## 🎯 Overview

This application leverages state-of-the-art deep learning to classify brain MRI scans and detect tumors with high accuracy. It provides medical professionals and researchers with:

- **Instant Classification**: Real-time tumor detection across 4 categories
- **Visual Explanations**: Grad-CAM heatmaps showing which regions the model focuses on
- **User-Friendly Interface**: Clean, intuitive web UI for seamless image analysis
- **GPU Optimized**: CUDA support for lightning-fast inference

---

## 📑 Table of Contents

1. [Overview](#-overview)
2. [Features](#-features)
   - [Core Capabilities](#-core-capabilities)
   - [Technical Highlights](#-technical-highlights)
3. [Tech Stack](#-tech-stack)
4. [Requirements](#-requirements)
5. [Quick Start](#-quick-start)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
6. [Usage](#-usage)
   - [Web Interface](#web-interface)
7. [How It Works](#-how-it-works)
   - [Architecture Flow](#architecture-flow)
   - [Model Details](#model-details)
   - [Classification Categories](#classification-categories)
8. [Grad-CAM Visualization](#-grad-cam-visualization)
9. [Performance Metrics](#-performance-metrics)
10. [Project Structure](#-project-structure)
11. [Demo](#-demo)
    - [Example Usage](#example-usage)
    - [Expected Output](#expected-output)
12. [Configuration](#-configuration)
13. [Deployment](#-deployment-future-work)
14. [Contributing](#-contributing)
15. [Disclaimer](#️-disclaimer)
16. [License](#-license)
17. [Author](#-author)
18. [Support](#-support)

---

## ✨ Features

### 🎨 Core Capabilities
- ✅ **4-Class Classification**: Detects Glioma, Meningioma, Pituitary tumors, or No tumor
- ✅ **Grad-CAM Visualization**: Visual explanations of model predictions with heatmaps
- ✅ **Real-time Processing**: Get results in seconds, not minutes
- ✅ **Batch Processing Ready**: Process multiple images sequentially

### 🚀 Technical Highlights
- ✅ **ResNet18 Architecture**: Proven deep learning model for image classification
- ✅ **PyTorch Framework**: Industry-standard ML framework
- ✅ **GPU Acceleration**: CUDA support for 10x faster inference
- ✅ **Responsive Design**: Works seamlessly on desktop and mobile

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Flask |
| **Deep Learning** | PyTorch, TorchVision |
| **Computer Vision** | OpenCV, PIL |
| **Visualization** | Matplotlib, NumPy |
| **Frontend** | HTML5, CSS3, JavaScript |

---

## 📋 Requirements

```
Flask
PyTorch
TorchVision
TorchAudio
OpenCV
Pillow
Matplotlib
NumPy
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip or conda
- Git LFS (for model file)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/Likith-2004/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
```

**2. Install Git LFS** (for model file)
```bash
git lfs install
git lfs pull
```

**3. Create virtual environment**
```bash
# Windows
python -m venv brainenv
brainenv\Scripts\activate

# macOS/Linux
python3 -m venv brainenv
source brainenv/bin/activate
```

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

**5. Run the application**
```bash
python app.py
```

**6. Access the web interface**
```
Open your browser and navigate to:
http://localhost:5000
```

---

## 💻 Usage

### Web Interface

1. **Upload MRI Image**
   - Click the upload button
   - Select a brain MRI scan (JPG, PNG)
   - Supported size: Any resolution (auto-resized to 224×224)

2. **View Results**
   - Classification result with confidence score
   - Grad-CAM heatmap showing model focus areas
   - Original image for comparison

3. **Interpret Predictions**
   - Red regions: Areas the model focused on
   - Intensity: How important that region was for the decision

---

## 🧠 How It Works

### Architecture Flow

```
Input MRI Image
       ↓
Preprocessing (224×224 resize, normalization)
       ↓
ResNet18 Feature Extraction
       ↓
Classification Layer
       ↓
Prediction + Confidence Score
       ↓
Grad-CAM Heatmap Generation
       ↓
Visualized Output
```

### Model Details

**Base Model**: ResNet18 (Pre-trained on ImageNet)
- **Input Size**: 224 × 224 pixels
- **Output Classes**: 4 (Glioma, Meningioma, Pituitary, No Tumor)
- **Inference Speed**: ~50-100ms per image (GPU)
- **Model Size**: 44 MB (stored via Git LFS)

### Classification Categories

| Class | Description | Characteristics |
|-------|-------------|-----------------|
| **Glioma** | Tumor from glial cells | Often fast-growing |
| **Meningioma** | Tumor from brain membranes | Usually benign |
| **Pituitary** | Tumor in pituitary gland | Hormonal impacts |
| **No Tumor** | Healthy brain scan | Normal MRI |

---

## 🔍 Grad-CAM Visualization

**What is Grad-CAM?**

Gradient-weighted Class Activation Mapping provides visual explanations for neural network predictions by:

1. Computing gradients of the target class w.r.t. feature maps
2. Creating a heatmap showing important regions
3. Overlaying on original image for interpretability

**Why it matters:**
- 🔬 **Medical Validation**: Doctors can verify model reasoning
- 🎓 **Explainability**: Understand what the AI "sees"
- 🏥 **Clinical Trust**: Increases confidence in AI-assisted diagnosis

---

## 📊 Performance Metrics

- **Accuracy**: Trained on comprehensive brain MRI dataset
- **Inference Time**: ~0.05-0.1s per image (GPU)
- **Memory Usage**: ~1.2 GB (model + dependencies)
- **GPU Memory**: ~2.5 GB during inference

---

## 📁 Project Structure

```
Brain-Tumor-Detection/
├── app.py                          # Flask application
├── utils.py                        # Grad-CAM utilities
├── requirements.txt                # Dependencies
├── model/
│   └── resnet18_brain_tumor.pth   # Trained model (Git LFS)
├── static/
│   ├── uploads/                   # User uploaded images
│   └── gradcam/                   # Generated heatmaps
├── templates/
│   └── index.html                 # Web interface
└── README.md
```

---

## 🎮 Demo

### Example Usage

```python
# 1. Upload image via web interface
# 2. Application processes the MRI scan
# 3. Displays classification result
# 4. Shows Grad-CAM visualization
# 5. Confidence scores for each class
```

### Expected Output

```
🔍 Prediction: Glioma Tumor
📊 Confidence: 94.2%

Heatmap: [Visualized on original image]
- Red regions show tumor location focus
- Blue regions show irrelevant areas
```

---

## ⚙️ Configuration

Edit `app.py` to customize:

```python
# Model path
MODEL_PATH = "model/resnet18_brain_tumor.pth"

# Upload folders
UPLOAD_FOLDER = "static/uploads"
GRADCAM_FOLDER = "static/gradcam"

# Class names
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Device (auto-detects GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## 🚀 Deployment (Future Work)

### Docker (Coming Soon)
```bash
docker build -t brain-tumor-app .
docker run -p 5000:5000 brain-tumor-app
```

### Cloud Deployment
- **AWS**: Deploy on EC2 or SageMaker
- **Google Cloud**: Use Vertex AI or App Engine
- **Heroku**: Simple one-click deployment
- **Local Server**: Run on any machine with Python 3.10+

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Ideas for Contribution
- 🎨 Improve UI/UX
- 📈 Enhance model accuracy with new training data
- 🔍 Add more visualization techniques
- 📱 Build mobile app wrapper
- 🚀 Optimize inference speed
- 📚 Add comprehensive documentation

---

## ⚠️ Disclaimer

**Important**: This application is designed for **research and educational purposes**.

⚠️ **NOT FOR MEDICAL DIAGNOSIS** — This tool should NOT be used for medical diagnosis without professional medical review. Always consult qualified radiologists and medical professionals for actual medical decisions.

The predictions are AI-based and may contain errors. Professional medical expertise is essential.

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Likith V K**
Deep Learning Enthusiast | AI Developer

[GitHub](https://github.com/Likith-2004) • [LinkedIn](https://linkedin.com/in/likith-vk-929b5b280)

---

## 📞 Support

Have questions or issues?

- 📧 **Email**: likithvk2004@gmail.com

---

<div align="center">

**⭐ If this project helped you, please give it a star!**

Made with ❤️ by [Likith](https://github.com/Likith-2004)

</div>
