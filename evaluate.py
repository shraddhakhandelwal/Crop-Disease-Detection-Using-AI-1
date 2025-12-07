"""
Evaluation script for trained model.

Usage:
    python evaluate.py --model models/saved_models/best_model.h5 --data_dir data/test
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import json
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import ImageDataLoader
from src.inference import DiseasePredictor, visualize_gradcam
from src.utils import setup_logger
from src.utils.visualization import (
    plot_confusion_matrix, generate_classification_report,
    plot_sample_predictions, plot_prediction_confidence
)

logger = setup_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--class_names', type=str,
                       default='models/saved_models/class_names.json',
                       help='Path to class names JSON')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                       help='Directory to save evaluation results')
    parser.add_argument('--gradcam', action='store_true',
                       help='Generate Grad-CAM visualizations')
    parser.add_argument('--n_gradcam_samples', type=int, default=10,
                       help='Number of Grad-CAM samples to generate')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load class names
    logger.info(f"Loading class names from {args.class_names}")
    with open(args.class_names, 'r') as f:
        class_names = json.load(f)
    
    logger.info(f"Found {len(class_names)} classes")
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    model = keras.models.load_model(args.model)
    
    # Load test data
    logger.info(f"Loading test data from {args.data_dir}")
    data_loader = ImageDataLoader(args.data_dir, image_size=(224, 224))
    
    images, labels, file_paths = data_loader.load_dataset()
    logger.info(f"Loaded {len(images)} test images")
    
    # Normalize with ImageNet statistics
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images_normalized = (images - mean) / std
    
    # Make predictions
    logger.info("Making predictions...")
    y_pred_proba = model.predict(images_normalized, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Convert labels to categorical
    y_true_cat = to_categorical(labels, len(class_names))
    
    # Evaluate metrics
    logger.info("Evaluating metrics...")
    results = model.evaluate(images_normalized, y_true_cat, verbose=1)
    metrics_dict = dict(zip(model.metrics_names, results))
    
    logger.info("\nEvaluation Metrics:")
    for metric, value in metrics_dict.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    # Confusion matrix
    logger.info("Generating confusion matrix...")
    plot_confusion_matrix(
        labels, y_pred,
        class_names,
        save_path=str(output_dir / 'confusion_matrix.png')
    )
    
    # Classification report
    logger.info("Generating classification report...")
    generate_classification_report(
        labels, y_pred,
        class_names,
        save_path=str(output_dir / 'classification_report.json')
    )
    
    # Confidence distribution
    logger.info("Plotting confidence distribution...")
    predictions = [
        {'confidence': float(y_pred_proba[i, y_pred[i]])}
        for i in range(len(y_pred))
    ]
    plot_prediction_confidence(
        predictions,
        save_path=str(output_dir / 'confidence_distribution.png')
    )
    
    # Sample predictions
    logger.info("Plotting sample predictions...")
    n_samples = min(16, len(images))
    sample_indices = np.random.choice(len(images), n_samples, replace=False)
    
    plot_sample_predictions(
        images[sample_indices],
        [class_names[labels[i]] for i in sample_indices],
        [class_names[y_pred[i]] for i in sample_indices],
        [y_pred_proba[i, y_pred[i]] for i in sample_indices],
        save_path=str(output_dir / 'sample_predictions.png'),
        n_samples=n_samples
    )
    
    # Grad-CAM visualizations
    if args.gradcam:
        logger.info("Generating Grad-CAM visualizations...")
        gradcam_dir = output_dir / 'gradcam'
        gradcam_dir.mkdir(exist_ok=True)
        
        # Select random samples
        n_samples = min(args.n_gradcam_samples, len(file_paths))
        sample_indices = np.random.choice(len(file_paths), n_samples, replace=False)
        
        for i, idx in enumerate(sample_indices):
            img_path = file_paths[idx]
            true_class = class_names[labels[idx]]
            pred_class = class_names[y_pred[idx]]
            
            save_path = gradcam_dir / f'gradcam_{i}_{true_class}_pred_{pred_class}.png'
            
            try:
                visualize_gradcam(
                    model,
                    img_path,
                    save_path=str(save_path),
                    image_size=(224, 224)
                )
            except Exception as e:
                logger.warning(f"Failed to generate Grad-CAM for {img_path}: {e}")
    
    logger.info("=" * 80)
    logger.info("EVALUATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
