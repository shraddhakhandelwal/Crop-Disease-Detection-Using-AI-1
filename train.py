"""
Main training script for crop disease detection model.

Usage:
    python train.py --config config/config.yaml --data_dir data/raw
"""
import argparse
import sys
from pathlib import Path
import numpy as np
from tensorflow.keras.utils import to_categorical

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, setup_logger, get_experiment_name
from src.preprocessing import ImageDataLoader
from src.model import DiseaseClassifier, TrainingManager
from src.utils.visualization import (
    plot_training_history, plot_confusion_matrix,
    plot_class_distribution, generate_classification_report
)

logger = setup_logger(__name__, 'logs/training.log')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train crop disease detection model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Path to dataset directory')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for this experiment')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model architecture (overrides config)')
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(args.config)
    
    # Override config with command line args
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.model_name:
        config['model']['name'] = args.model_name
    
    # Generate experiment name
    experiment_name = args.experiment_name or get_experiment_name()
    logger.info(f"Starting experiment: {experiment_name}")
    
    # Step 1: Load and prepare data
    logger.info("=" * 80)
    logger.info("STEP 1: Loading and preparing data")
    logger.info("=" * 80)
    
    data_loader = ImageDataLoader(
        data_dir=args.data_dir,
        image_size=tuple(config['data']['image_size'])
    )
    
    if not data_loader.class_names:
        logger.error(f"No classes found in {args.data_dir}")
        logger.error("Please ensure your data is organized in class subdirectories")
        return
    
    logger.info(f"Found {len(data_loader.class_names)} classes")
    
    # Load dataset
    logger.info("Loading images...")
    images, labels, file_paths = data_loader.load_dataset()
    
    logger.info(f"Loaded {len(images)} images")
    
    # Plot class distribution
    plot_class_distribution(
        labels,
        data_loader.class_names,
        save_path=f"results/{experiment_name}_class_distribution.png",
        title="Training Data Class Distribution"
    )
    
    # Split data
    logger.info("Splitting data into train/val/test...")
    data_splits = data_loader.split_data(
        images, labels,
        val_split=config['data']['validation_split'],
        test_split=config['data']['test_split'],
        random_state=config['data']['seed']
    )
    
    X_train, y_train = data_splits['train']
    X_val, y_val = data_splits['val']
    X_test, y_test = data_splits['test']
    
    # Convert labels to one-hot encoding
    num_classes = len(data_loader.class_names)
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    # Step 2: Build model
    logger.info("=" * 80)
    logger.info("STEP 2: Building model")
    logger.info("=" * 80)
    
    classifier = DiseaseClassifier(
        model_name=config['model']['name'],
        num_classes=num_classes,
        input_shape=config['model']['input_shape'],
        pretrained=config['model']['pretrained'],
        dropout_rate=config['model']['dropout_rate']
    )
    
    model = classifier.build_model()
    
    # Compile model
    model = DiseaseClassifier.compile_model(
        model,
        learning_rate=config['training']['learning_rate'],
        optimizer=config['training']['optimizer'],
        loss=config['training']['loss']
    )
    
    logger.info("\n" + classifier.get_model_summary(model))
    
    # Step 3: Train model
    logger.info("=" * 80)
    logger.info("STEP 3: Training model")
    logger.info("=" * 80)
    
    trainer = TrainingManager(config, experiment_name)
    
    # Compute class weights if enabled
    class_weights = None
    if config.get('class_weights', {}).get('enabled', True):
        logger.info("Computing class weights for imbalanced data...")
        class_weights = trainer.compute_class_weights(y_train)
    
    # Train
    history = trainer.train_model(
        model,
        train_data=(X_train, y_train_cat),
        val_data=(X_val, y_val_cat),
        class_weights=class_weights
    )
    
    # Save training history
    history_path = trainer.save_training_history(history)
    
    # Plot training curves
    plot_training_history(
        history_path,
        save_path=f"results/{experiment_name}_training_history.png",
        metrics=['accuracy', 'loss']
    )
    
    # Step 4: Evaluate model
    logger.info("=" * 80)
    logger.info("STEP 4: Evaluating model")
    logger.info("=" * 80)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_loss, test_acc, test_prec, test_rec, test_auc = model.evaluate(
        X_test, y_test_cat, verbose=1
    )
    
    logger.info(f"\nTest Results:")
    logger.info(f"  Loss: {test_loss:.4f}")
    logger.info(f"  Accuracy: {test_acc:.4f}")
    logger.info(f"  Precision: {test_prec:.4f}")
    logger.info(f"  Recall: {test_rec:.4f}")
    logger.info(f"  AUC: {test_auc:.4f}")
    
    # Generate predictions
    logger.info("Generating predictions for analysis...")
    y_pred_proba = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Confusion matrix
    plot_confusion_matrix(
        y_test, y_pred,
        data_loader.class_names,
        save_path=f"results/{experiment_name}_confusion_matrix.png"
    )
    
    # Classification report
    generate_classification_report(
        y_test, y_pred,
        data_loader.class_names,
        save_path=f"results/{experiment_name}_classification_report.json"
    )
    
    # Step 5: Save final model
    logger.info("=" * 80)
    logger.info("STEP 5: Saving model")
    logger.info("=" * 80)
    
    model_path = trainer.save_final_model(model, f'{experiment_name}_final.h5')
    
    # Save class names for inference
    import json
    class_names_path = Path(config['paths']['saved_model_dir']) / 'class_names.json'
    with open(class_names_path, 'w') as f:
        json.dump(data_loader.class_names, f, indent=2)
    logger.info(f"Saved class names to {class_names_path}")
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Results saved to: results/{experiment_name}_*")
    logger.info(f"Logs saved to: {trainer.logs_dir}")


if __name__ == '__main__':
    main()
