"""
Inference script for making predictions on new images.

Usage:
    python predict.py --model models/saved_models/best_model.h5 --image path/to/image.jpg
    python predict.py --model models/saved_models/best_model.h5 --image_dir path/to/images/
"""
import argparse
import sys
from pathlib import Path
import json
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.inference import DiseasePredictor, visualize_gradcam
from src.utils import setup_logger

logger = setup_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make predictions on crop images')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--class_names', type=str,
                       default='models/saved_models/class_names.json',
                       help='Path to class names JSON')
    parser.add_argument('--image', type=str,
                       help='Path to single image')
    parser.add_argument('--image_dir', type=str,
                       help='Path to directory of images')
    parser.add_argument('--output', type=str, default='predictions.json',
                       help='Path to save predictions JSON')
    parser.add_argument('--confidence_threshold', type=float, default=0.7,
                       help='Confidence threshold for predictions')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Number of top predictions to show')
    parser.add_argument('--gradcam', action='store_true',
                       help='Generate Grad-CAM visualizations')
    parser.add_argument('--gradcam_dir', type=str, default='results/gradcam',
                       help='Directory to save Grad-CAM images')
    
    return parser.parse_args()


def main():
    """Main prediction function."""
    args = parse_args()
    
    # Validate inputs
    if not args.image and not args.image_dir:
        logger.error("Must specify either --image or --image_dir")
        return
    
    # Load class names
    logger.info(f"Loading class names from {args.class_names}")
    with open(args.class_names, 'r') as f:
        class_names = json.load(f)
    
    # Initialize predictor
    logger.info(f"Loading model from {args.model}")
    predictor = DiseasePredictor(
        model_path=args.model,
        class_names=class_names,
        confidence_threshold=args.confidence_threshold
    )
    
    # Get image paths
    image_paths = []
    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_paths.extend(image_dir.glob(ext))
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    # Make predictions
    all_predictions = []
    
    for img_path in image_paths:
        logger.info(f"\nProcessing: {img_path}")
        
        try:
            # Predict
            result = predictor.predict(str(img_path), top_k=args.top_k)
            
            # Add file path to result
            result['image_path'] = str(img_path)
            all_predictions.append(result)
            
            # Display results
            logger.info(f"  Predicted: {result['predicted_class']}")
            logger.info(f"  Confidence: {result['confidence']:.2%}")
            logger.info(f"  Confident: {result['is_confident']}")
            
            if args.top_k > 1:
                logger.info(f"  Top {args.top_k} predictions:")
                for i, pred in enumerate(result['top_predictions'], 1):
                    logger.info(f"    {i}. {pred['class']}: {pred['confidence']:.2%}")
            
            # Generate Grad-CAM if requested
            if args.gradcam:
                gradcam_dir = Path(args.gradcam_dir)
                gradcam_dir.mkdir(parents=True, exist_ok=True)
                
                img_name = Path(img_path).stem
                pred_class = result['predicted_class'].replace(' ', '_')
                gradcam_path = gradcam_dir / f'{img_name}_{pred_class}_gradcam.png'
                
                logger.info(f"  Generating Grad-CAM to {gradcam_path}")
                visualize_gradcam(
                    predictor.model,
                    str(img_path),
                    save_path=str(gradcam_path)
                )
        
        except Exception as e:
            logger.error(f"  Error processing {img_path}: {e}")
            continue
    
    # Save predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    
    logger.info(f"\nSaved predictions to {output_path}")
    
    # Summary statistics
    if all_predictions:
        confident_count = sum(1 for p in all_predictions if p['is_confident'])
        avg_confidence = np.mean([p['confidence'] for p in all_predictions])
        
        logger.info("\n" + "=" * 80)
        logger.info("PREDICTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total images: {len(all_predictions)}")
        logger.info(f"Confident predictions: {confident_count} ({confident_count/len(all_predictions):.1%})")
        logger.info(f"Average confidence: {avg_confidence:.2%}")
        
        # Most common predictions
        from collections import Counter
        pred_counts = Counter(p['predicted_class'] for p in all_predictions)
        logger.info("\nMost common predictions:")
        for disease, count in pred_counts.most_common(5):
            logger.info(f"  {disease}: {count}")


if __name__ == '__main__':
    main()
