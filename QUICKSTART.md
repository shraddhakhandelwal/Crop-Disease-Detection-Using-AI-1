# Quick Start Guide - Crop Disease Detection AI

This guide will get you up and running with the crop disease detection system in 5 minutes.

## 🚀 Fastest Way: Web Interface

```bash
# 1. Clone and install
git clone https://github.com/shraddhakhandelwal/Crop-Disease-Detection-Using-AI-1.git
cd Crop-Disease-Detection-Using-AI-1
pip install -r requirements.txt

# 2. Start web server
python api/app.py

# 3. Open browser
# Navigate to: http://localhost:5000
```

**That's it!** You now have a fully functional web interface where you can:
- 📤 Upload leaf images (drag & drop)
- 🔍 Get instant disease predictions
- 📊 View confidence scores
- 🎨 See Grad-CAM visualizations

*Note: For best results, train your own model first (see below) or use a pre-trained model.*

---

## 📋 Full Setup Guide

## Prerequisites
- Python 3.10+
- pip or conda
- (Optional) NVIDIA GPU for faster training

## Step 1: Setup (2 minutes)

```bash
# Clone repository
git clone https://github.com/shraddhakhandelwal/Crop-Disease-Detection-Using-AI-1.git
cd Crop-Disease-Detection-Using-AI-1

# Run automated setup
bash setup.sh

# Or manual setup:
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 2: Get Dataset (3 minutes)

### Option A: Automated Download (Kaggle API)
```bash
# Install Kaggle
pip install kaggle

# Setup API key from https://www.kaggle.com/account
mkdir -p ~/.kaggle
# Place your kaggle.json in ~/.kaggle/

# Download dataset
bash scripts/download_data.sh
```

### Option B: Manual Download
1. Visit: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
2. Download ZIP file
3. Extract to `data/raw/`

Expected structure:
```
data/raw/
├── Apple___Apple_scab/
├── Apple___Black_rot/
├── Tomato___Bacterial_spot/
└── ... (38 classes total)
```

## Step 3: Train Your First Model (30-60 minutes)

```bash
# Quick training (small dataset sample, for testing)
python train.py --data_dir data/raw --epochs 5

# Full training (recommended)
python train.py --data_dir data/raw --epochs 50 --model_name efficientnet_b0
```

**What's happening:**
- Loading and preprocessing images
- Training with data augmentation
- Automatic checkpointing
- Generating performance visualizations

**Outputs:**
- Model: `models/checkpoints/<timestamp>/best_model.h5`
- Metrics: `results/<timestamp>_*.png`
- Logs: `logs/<timestamp>/`

## Step 4: Test Predictions

```bash
# Single image prediction
python predict.py \
    --model models/saved_models/best_model.h5 \
    --image data/samples/leaf.jpg \
    --top_k 3

# Batch prediction
python predict.py \
    --model models/saved_models/best_model.h5 \
    --image_dir data/samples/ \
    --output predictions.json
```

## Step 5: Start API Server (Optional)

```bash
# Local
python api/app.py

# Docker
docker-compose up -d
```

Test API:
```bash
# Health check
curl http://localhost:5000/health

# Predict
curl -X POST -F "image=@leaf.jpg" http://localhost:5000/predict
```

## Common Commands Cheat Sheet

```bash
# Training
python train.py --data_dir data/raw --epochs 50

# Evaluation
python evaluate.py --model models/saved_models/best_model.h5 --data_dir data/test

# Prediction
python predict.py --model models/saved_models/best_model.h5 --image leaf.jpg

# API
python api/app.py

# Tests
pytest tests/ -v

# TensorBoard
tensorboard --logdir logs/
```

## Troubleshooting

### "No module named 'src'"
```bash
# Ensure you're in project root
cd /path/to/Crop-Disease-Detection-Using-AI-1
python train.py ...
```

### "Out of memory" during training
```bash
# Edit config/config.yaml
# Reduce batch_size from 32 to 16 or 8
```

### "Model file not found"
```bash
# Check models directory
ls -la models/saved_models/

# If empty, train a model first
python train.py --data_dir data/raw
```

## Next Steps

1. **Experiment with models**: Try different architectures in config.yaml
2. **Fine-tune**: Adjust hyperparameters for better performance
3. **Deploy**: Use Docker for production deployment
4. **Custom dataset**: Add your own disease images to data/raw/

## Getting Help

- **Documentation**: See README.md for full details
- **Issues**: https://github.com/shraddhakhandelwal/Crop-Disease-Detection-Using-AI-1/issues
- **Config**: All settings in config/config.yaml

## Typical Training Output

```
Loading configuration...
Starting experiment: exp_20250107_143022

STEP 1: Loading and preparing data
Found 38 classes
Loading images...
Loaded 54305 images
Splitting data into train/val/test...
Data split - Train: 38013, Val: 10861, Test: 5431

STEP 2: Building model
Initializing efficientnet_b0 with 38 classes
Built model with 4,103,122 total parameters
Trainable parameters: 1,290,350

STEP 3: Training model
Computing class weights for imbalanced data...
Starting training...
Epoch 1/50
1188/1188 [==============================] - 156s 131ms/step - loss: 0.4235 - accuracy: 0.8821
Epoch 2/50
1188/1188 [==============================] - 152s 128ms/step - loss: 0.1523 - accuracy: 0.9543
...
```

Happy disease detection! 🌱🔬
