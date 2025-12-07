# Crop Disease Detection Using AI 🌱🔬

An AI-powered crop disease detection system using deep learning to help farmers identify plant diseases early and optimize pesticide use. Built with TensorFlow/Keras and transfer learning for accurate, production-ready disease classification.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Features

- **Transfer Learning**: Pre-trained models (EfficientNet, ResNet, MobileNet) for high accuracy
- **38 Disease Classes**: Comprehensive coverage of common crop diseases
- **Data Augmentation**: Advanced augmentation pipeline for robust training
- **Model Interpretability**: Grad-CAM visualizations for explainable AI
- **REST API**: Flask-based API for easy integration
- **Docker Support**: Containerized deployment ready
- **Comprehensive Metrics**: Confusion matrices, classification reports, ROC curves
- **Production Ready**: Logging, error handling, configuration management

## 📁 Project Structure

```
├── api/                    # Flask REST API
│   ├── app.py             # API server
│   └── client_example.py  # API client example
├── config/                # Configuration files
│   └── config.yaml        # Main configuration
├── data/                  # Dataset storage
│   ├── raw/              # Raw images (organized by class)
│   ├── processed/        # Processed data
│   └── samples/          # Sample images
├── models/               # Model storage
│   ├── checkpoints/     # Training checkpoints
│   └── saved_models/    # Final trained models
├── notebooks/           # Jupyter notebooks (examples)
├── scripts/             # Utility scripts
│   └── download_data.sh # Dataset download helper
├── src/                 # Source code
│   ├── inference/      # Prediction and Grad-CAM
│   ├── model/          # Model architectures
│   ├── preprocessing/  # Data loading and augmentation
│   └── utils/          # Utilities (config, logging, viz)
├── tests/              # Unit tests
├── train.py            # Training script
├── evaluate.py         # Evaluation script
├── predict.py          # Inference script
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker configuration
└── docker-compose.yml  # Docker Compose setup
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/shraddhakhandelwal/Crop-Disease-Detection-Using-AI-1.git
cd Crop-Disease-Detection-Using-AI-1

# Run setup script (creates venv, installs dependencies)
bash setup.sh

# Or manually:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Dataset

**PlantVillage Dataset** (38 classes, ~54,000 images):

```bash
# Option 1: Automated download (requires Kaggle API)
bash scripts/download_data.sh

# Option 2: Manual download
# 1. Visit: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
# 2. Download and extract to data/raw/
```

Expected structure:
```
data/raw/
├── Tomato___Bacterial_spot/
├── Potato___Early_blight/
├── Corn_(maize)___Common_rust/
└── ...
```

### 3. Train Model

```bash
# Basic training
python train.py --data_dir data/raw

# Custom configuration
python train.py \
    --data_dir data/raw \
    --epochs 50 \
    --batch_size 32 \
    --model_name efficientnet_b0 \
    --experiment_name my_experiment
```

**Available models**: `efficientnet_b0`, `resnet50`, `mobilenet_v2`, `vgg16`, `inceptionv3`

Training outputs:
- Model checkpoints: `models/checkpoints/`
- Final model: `models/saved_models/`
- Training logs: `logs/`
- Visualizations: `results/`

### 4. Evaluate Model

```bash
python evaluate.py \
    --model models/saved_models/best_model.h5 \
    --data_dir data/test \
    --gradcam  # Optional: generate Grad-CAM visualizations
```

### 5. Make Predictions

```bash
# Single image
python predict.py \
    --model models/saved_models/best_model.h5 \
    --image path/to/leaf.jpg \
    --gradcam

# Batch prediction
python predict.py \
    --model models/saved_models/best_model.h5 \
    --image_dir path/to/images/ \
    --output predictions.json
```

## 🌐 REST API

### Start API Server

```bash
# Local
python api/app.py

# Docker
docker-compose up --build
```

### API Endpoints

**Health Check**
```bash
curl http://localhost:5000/health
```

**Get Classes**
```bash
curl http://localhost:5000/classes
```

**Predict Disease**
```bash
curl -X POST -F "image=@leaf.jpg" \
     -F "top_k=3" \
     http://localhost:5000/predict
```

**Example Response:**
```json
{
  "success": true,
  "predicted_class": "Tomato___Late_blight",
  "confidence": 0.945,
  "is_confident": true,
  "top_predictions": [
    {"class": "Tomato___Late_blight", "confidence": 0.945},
    {"class": "Tomato___Early_blight", "confidence": 0.032},
    {"class": "Tomato___Leaf_Mold", "confidence": 0.015}
  ]
}
```

### Python Client Example

```python
import requests

response = requests.post(
    'http://localhost:5000/predict',
    files={'image': open('leaf.jpg', 'rb')},
    data={'top_k': 3}
)

result = response.json()
print(f"Disease: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

```yaml
data:
  image_size: [224, 224]
  batch_size: 32
  validation_split: 0.2

model:
  name: "efficientnet_b0"
  num_classes: 38
  dropout_rate: 0.3

training:
  epochs: 50
  learning_rate: 0.001
  optimizer: "adam"
  early_stopping_patience: 10

augmentation:
  rotation_range: 20
  horizontal_flip: true
  zoom_range: 0.2
  brightness_range: [0.8, 1.2]
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_model.py -v
```

## 📊 Model Performance

Example results on PlantVillage dataset:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| EfficientNetB0 | 98.5% | 98.3% | 98.2% | 98.2% |
| ResNet50 | 97.8% | 97.6% | 97.5% | 97.5% |
| MobileNetV2 | 96.2% | 96.0% | 95.9% | 95.9% |

## 🔬 Technologies Used

- **Deep Learning**: TensorFlow 2.15, Keras
- **Computer Vision**: OpenCV, Albumentations
- **Data Science**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **API**: Flask, Flask-CORS
- **Deployment**: Docker, Docker Compose
- **Experiment Tracking**: MLflow, TensorBoard
- **Testing**: Pytest

## 📚 Key Concepts

### Transfer Learning
Uses pre-trained ImageNet models as feature extractors, fine-tuned for crop disease classification.

### Data Augmentation
- Geometric: Rotation, flipping, shifting, zooming
- Color: Brightness, contrast, saturation adjustments
- Noise: Gaussian noise, blur

### Grad-CAM Visualization
Generates heatmaps showing which parts of the image the model focuses on for predictions.

### Class Imbalance Handling
Automatic class weight computation to handle uneven disease distribution in datasets.

## 🛠️ Development Workflow

```bash
# 1. Prepare data
# Organize images in class subdirectories under data/raw/

# 2. Train model
python train.py --data_dir data/raw --epochs 50

# 3. Evaluate
python evaluate.py --model models/saved_models/best_model.h5 --data_dir data/test

# 4. Test API locally
python api/app.py

# 5. Deploy with Docker
docker-compose up -d
```

## 📈 Results & Outputs

Training generates:
- **Confusion Matrix**: `results/confusion_matrix.png`
- **Training History**: `results/training_history.png`
- **Class Distribution**: `results/class_distribution.png`
- **Classification Report**: `results/classification_report.json`
- **Grad-CAM Visualizations**: `results/gradcam/`
- **TensorBoard Logs**: `logs/tensorboard/`

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **PlantVillage Dataset**: Primary dataset for crop disease images
- **TensorFlow/Keras**: Deep learning framework
- **Agricultural Research**: Domain knowledge for disease classification

## 📧 Contact

**Author**: Shraddha Khandelwal
**Repository**: [github.com/shraddhakhandelwal/Crop-Disease-Detection-Using-AI-1](https://github.com/shraddhakhandelwal/Crop-Disease-Detection-Using-AI-1)

## 🗺️ Roadmap

- [ ] Mobile app integration
- [ ] Real-time video detection
- [ ] Multi-crop support expansion
- [ ] Treatment recommendations
- [ ] Progressive web app
- [ ] Edge device deployment (Raspberry Pi, mobile)

---

**Made with ❤️ for farmers and agricultural innovation**