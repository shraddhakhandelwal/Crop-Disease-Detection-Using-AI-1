"""
Quick demo training script to verify the complete pipeline.
Trains a small model for 3 epochs on sample data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import ImageDataLoader
from src.model import DiseaseClassifier
from src.utils import load_config, setup_logger
import numpy as np
from tensorflow.keras.utils import to_categorical

logger = setup_logger(__name__)

def quick_train():
    """Quick training demo."""
    logger.info("=" * 80)
    logger.info("QUICK TRAINING DEMO")
    logger.info("=" * 80)
    
    # Load data
    logger.info("\n1. Loading sample data...")
    loader = ImageDataLoader('data/raw', image_size=(224, 224))
    images, labels, _ = loader.load_dataset()
    logger.info(f"   ✓ Loaded {len(images)} images from {len(loader.class_names)} classes")
    
    # Split data
    logger.info("\n2. Splitting data...")
    splits = loader.split_data(images, labels, val_split=0.2, test_split=0.1)
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    
    # One-hot encode
    y_train_cat = to_categorical(y_train, len(loader.class_names))
    y_val_cat = to_categorical(y_val, len(loader.class_names))
    logger.info(f"   ✓ Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Build model
    logger.info("\n3. Building model...")
    classifier = DiseaseClassifier(
        model_name='mobilenet_v2',  # Smaller model for demo
        num_classes=len(loader.class_names),
        input_shape=(224, 224, 3),
        pretrained=True,
        dropout_rate=0.3
    )
    model = classifier.build_model()
    model = DiseaseClassifier.compile_model(model, learning_rate=0.001)
    logger.info(f"   ✓ Model ready with {model.count_params():,} parameters")
    
    # Train
    logger.info("\n4. Training for 3 epochs (demo)...")
    history = model.fit(
        X_train, y_train_cat,
        batch_size=16,
        epochs=3,
        validation_data=(X_val, y_val_cat),
        verbose=1
    )
    
    logger.info("\n5. Evaluating...")
    test_loss, test_acc = model.evaluate(X_val, y_val_cat, verbose=0)[:2]
    logger.info(f"   ✓ Validation Accuracy: {test_acc:.4f}")
    logger.info(f"   ✓ Validation Loss: {test_loss:.4f}")
    
    # Save model
    logger.info("\n6. Saving model...")
    Path('models/saved_models').mkdir(parents=True, exist_ok=True)
    model.save('models/saved_models/demo_model.h5')
    logger.info("   ✓ Model saved to models/saved_models/demo_model.h5")
    
    # Save class names
    import json
    with open('models/saved_models/class_names.json', 'w') as f:
        json.dump(loader.class_names, f, indent=2)
    logger.info("   ✓ Class names saved")
    
    logger.info("\n" + "=" * 80)
    logger.info("DEMO COMPLETE! ✅")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("  1. Make predictions: python predict.py --model models/saved_models/demo_model.h5 --image data/raw/Tomato___Healthy/Tomato___Healthy_000.jpg")
    logger.info("  2. Start API: python api/app.py")
    logger.info("  3. Full training: python train.py --data_dir data/raw --epochs 20")

if __name__ == '__main__':
    quick_train()
