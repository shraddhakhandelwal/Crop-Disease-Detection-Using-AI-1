# Crop Disease Detection AI - Developer Guide

## Project Overview
Production-ready AI system for crop disease detection using deep learning. Built with TensorFlow/Keras and transfer learning, targeting 38 plant disease classes from the PlantVillage dataset.

## Architecture Overview

### Core Components
1. **Data Pipeline** (`src/preprocessing/`)
   - `data_loader.py`: ImageDataLoader handles image loading, resizing, normalization (ImageNet stats)
   - `augmentation.py`: Albumentations-based augmentation (rotation, flip, brightness, noise, blur)
   - Data split: 70% train, 20% val, 10% test with stratification

2. **Model Layer** (`src/model/`)
   - `architecture.py`: Transfer learning with EfficientNet/ResNet/MobileNet backbones
   - Pattern: Frozen base → Dense(512) → BatchNorm → Dropout → Dense(256) → Output
   - `trainer.py`: TrainingManager with ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

3. **Inference Engine** (`src/inference/`)
   - `predictor.py`: DiseasePredictor for single/batch prediction with confidence thresholds
   - `gradcam.py`: Grad-CAM visualization on last conv layer for model interpretability
   - EnsemblePredictor for multi-model weighted averaging

4. **API Layer** (`api/app.py`)
   - Flask REST API: `/predict`, `/predict_batch`, `/classes`, `/health`
   - Handles image upload, preprocessing, prediction, JSON response
   - Docker-ready with gunicorn for production

### Configuration System
All settings centralized in `config/config.yaml`:
- Data: image_size=[224,224], batch_size, splits, seed
- Model: architecture name, num_classes=38, dropout_rate
- Training: epochs, lr, optimizer, early stopping patience
- Augmentation: rotation, flip, zoom, brightness ranges
- Inference: confidence_threshold, top_k predictions

## Development Workflows

### Training a New Model
```bash
# Standard training
python train.py --data_dir data/raw --epochs 50 --model_name efficientnet_b0

# With custom config
python train.py --config custom_config.yaml --experiment_name exp_001
```

**What happens internally:**
1. ImageDataLoader scans data_dir for class subdirectories
2. Loads images → normalize to [0,1] → apply ImageNet normalization
3. Stratified split into train/val/test
4. DiseaseClassifier builds model: base (frozen) + custom head
5. TrainingManager creates callbacks, computes class weights
6. Training with data augmentation on-the-fly
7. Saves: best_model.h5, checkpoints, training_log.csv, history plots

### Evaluation & Analysis
```bash
python evaluate.py --model models/saved_models/best_model.h5 --data_dir data/test --gradcam
```

Generates: confusion_matrix.png, classification_report.json, confidence_distribution.png, sample_predictions.png, gradcam visualizations

### Making Predictions
```bash
# Single image
python predict.py --model models/saved_models/best_model.h5 --image leaf.jpg --gradcam

# Batch with JSON output
python predict.py --model models/saved_models/best_model.h5 --image_dir images/ --output results.json
```

## Key Patterns & Conventions

### Data Handling
- **Directory structure**: `data/raw/{class_name}/*.jpg`
- **Normalization**: Images normalized with `mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]` (ImageNet)
- **Class weights**: Auto-computed with sklearn's `compute_class_weight('balanced')` for imbalanced data
- **Augmentation**: Applied during training via Albumentations, NOT baked into dataset

### Model Architecture
- **Transfer learning pattern**: Load pre-trained base → freeze → add custom head → train
- **Fine-tuning**: Optionally unfreeze top N layers of base model with `fine_tune_layers` parameter
- **Dropout strategy**: 0.3 after first dense, 0.3 after second, 0.15 before output
- **Output**: Softmax activation for multi-class classification

### Training Best Practices
- **Callbacks order matters**: ModelCheckpoint → EarlyStopping → ReduceLROnPlateau
- **Monitor metric**: `val_accuracy` for checkpoint, `val_loss` for LR reduction
- **Class weights**: Always use for PlantVillage (imbalanced classes)
- **Batch size**: 32 default, reduce if GPU memory limited

### Inference Pipeline
1. Load image → resize to (224,224) → normalize [0,1] → ImageNet normalize
2. Add batch dimension: `np.expand_dims(img, axis=0)`
3. Model.predict() → returns probabilities
4. Top-K selection with confidence threshold check
5. Optional: Generate Grad-CAM for visualization

### Logging & Debugging
- All modules use `setup_logger()` from `src/utils/logger.py`
- Format: `timestamp - module - level - message`
- Training logs: `logs/{experiment_name}/training_log.csv`
- TensorBoard: `tensorboard --logdir logs/{experiment_name}`

## Common Tasks

### Adding a New Model Architecture
1. Add to `MODEL_REGISTRY` in `src/model/architecture.py`
2. Ensure it follows Keras Applications API (include_top=False, pooling='avg')
3. Update config.yaml with new model name option

### Changing Augmentation
Edit `config/config.yaml` under `augmentation:` section. Changes apply immediately to next training run.

### Custom Dataset
1. Organize as `data/raw/{disease_class_name}/*.jpg`
2. Update `config.yaml`: `num_classes`, class names saved automatically during training
3. Run training: `python train.py --data_dir data/raw`

### API Deployment
```bash
# Local testing
python api/app.py

# Docker production
docker-compose up -d
# API available at http://localhost:5000
```

### Running Tests
```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific module
pytest tests/test_model.py::TestDiseaseClassifier -v
```

## File Organization

### Important Files
- `train.py`: Main training script (argparse CLI)
- `evaluate.py`: Model evaluation with metrics/visualizations
- `predict.py`: Inference on new images
- `config/config.yaml`: **Central configuration** - edit this for experiments
- `src/utils/config_loader.py`: YAML config parser
- `src/utils/visualization.py`: All plotting functions (confusion matrix, ROC, etc.)

### Model Storage
- Checkpoints during training: `models/checkpoints/{experiment_name}/`
- Best model: `models/checkpoints/{experiment_name}/best_model.h5`
- Final saved models: `models/saved_models/{experiment_name}_final.h5`
- Class names: `models/saved_models/class_names.json` (required for inference)

### Results & Artifacts
- Training plots: `results/{experiment_name}_*.png`
- Evaluation outputs: `results/evaluation/`
- Predictions: `predictions.json` or custom output path

## Troubleshooting

### Out of Memory (GPU)
- Reduce `batch_size` in config.yaml (try 16 or 8)
- Use smaller model: `mobilenet_v2` instead of `efficientnet_b0`
- Enable mixed precision: Add `tf.keras.mixed_precision.set_global_policy('mixed_float16')` in train.py

### Poor Model Performance
- Check class distribution: `plot_class_distribution()` shows imbalance
- Verify class weights are enabled in config
- Increase augmentation diversity
- Try different base model architecture
- Increase training epochs or adjust learning rate

### API Errors
- Ensure model file exists: `models/saved_models/*.h5`
- Check `class_names.json` is present
- Verify image format (JPEG/PNG, RGB)
- Check image preprocessing matches training normalization

## Integration Points

### MLflow Tracking
- Configured in config.yaml under `mlflow:`
- Training automatically logs metrics if MLflow enabled
- Access: `mlflow ui --backend-store-uri mlruns`

### TensorBoard
- Logs saved to `logs/{experiment_name}/`
- View: `tensorboard --logdir logs/`
- Shows: training curves, model graph, histograms

### External Datasets
- PlantVillage: 38 classes, ~54k images (recommended)
- Custom datasets: Must follow directory structure convention
- Download helper: `scripts/download_data.sh`

## Performance Considerations

### Training Speed
- GPU: ~2-3 min/epoch on modern GPU (NVIDIA T4/V100)
- CPU: ~20-30 min/epoch (not recommended for production)
- Data loading: Uses OpenCV (faster than PIL for large datasets)

### Inference Speed
- EfficientNetB0: ~50ms/image on GPU, ~200ms on CPU
- MobileNetV2: ~30ms/image on GPU, ~100ms on CPU (best for edge)
- Batch inference: 5-10x faster than individual predictions

### Model Size
- EfficientNetB0: ~20MB
- ResNet50: ~90MB  
- MobileNetV2: ~9MB (best for mobile deployment)

## Resources & References
- PlantVillage Dataset: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
- TensorFlow Models: https://www.tensorflow.org/api_docs/python/tf/keras/applications
- Grad-CAM Paper: https://arxiv.org/abs/1610.02391
