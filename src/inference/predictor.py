"""
Inference engine for crop disease prediction.
"""
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Union
import tensorflow as tf
from tensorflow import keras
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DiseasePredictor:
    """Handle disease prediction on crop images."""
    
    def __init__(
        self,
        model_path: str,
        class_names: List[str],
        image_size: Tuple[int, int] = (224, 224),
        confidence_threshold: float = 0.7
    ):
        """
        Initialize disease predictor.
        
        Args:
            model_path: Path to trained model file
            class_names: List of disease class names
            image_size: Expected input image size
            confidence_threshold: Minimum confidence for prediction
        """
        self.model_path = Path(model_path)
        self.class_names = class_names
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load trained model from file."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        logger.info(f"Loading model from {self.model_path}")
        self.model = keras.models.load_model(str(self.model_path))
        logger.info("Model loaded successfully")
    
    def preprocess_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """
        Preprocess image for prediction.
        
        Args:
            image: Image path or numpy array
            
        Returns:
            Preprocessed image array
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Failed to load image: {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()
        
        # Resize
        img = cv2.resize(img, self.image_size)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        # Normalize with ImageNet statistics
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        return img
    
    def predict(
        self,
        image: Union[str, np.ndarray],
        top_k: int = 3
    ) -> Dict[str, any]:
        """
        Predict disease from image.
        
        Args:
            image: Image path or numpy array
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        processed_img = self.preprocess_image(image)
        
        # Add batch dimension
        batch_img = np.expand_dims(processed_img, axis=0)
        
        # Predict
        predictions = self.model.predict(batch_img, verbose=0)[0]
        
        # Get top k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_predictions = [
            {
                'class': self.class_names[idx],
                'confidence': float(predictions[idx]),
                'class_index': int(idx)
            }
            for idx in top_indices
        ]
        
        # Primary prediction
        primary = top_predictions[0]
        
        # Determine if confident
        is_confident = primary['confidence'] >= self.confidence_threshold
        
        result = {
            'predicted_class': primary['class'],
            'confidence': primary['confidence'],
            'is_confident': is_confident,
            'top_predictions': top_predictions,
            'all_probabilities': predictions.tolist()
        }
        
        logger.info(
            f"Prediction: {primary['class']} "
            f"(confidence: {primary['confidence']:.2%})"
        )
        
        return result
    
    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        batch_size: int = 32
    ) -> List[Dict[str, any]]:
        """
        Predict diseases for multiple images.
        
        Args:
            images: List of image paths or arrays
            batch_size: Batch size for prediction
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Preprocess batch
            processed_batch = np.array([
                self.preprocess_image(img) for img in batch
            ])
            
            # Predict
            predictions = self.model.predict(processed_batch, verbose=0)
            
            # Process results
            for j, pred in enumerate(predictions):
                top_idx = np.argmax(pred)
                results.append({
                    'predicted_class': self.class_names[top_idx],
                    'confidence': float(pred[top_idx]),
                    'is_confident': pred[top_idx] >= self.confidence_threshold,
                })
        
        logger.info(f"Processed {len(images)} images in batch")
        
        return results
    
    def get_feature_vector(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """
        Extract feature vector from penultimate layer.
        
        Args:
            image: Image path or array
            
        Returns:
            Feature vector
        """
        # Create feature extractor model
        feature_model = keras.Model(
            inputs=self.model.input,
            outputs=self.model.layers[-2].output  # Penultimate layer
        )
        
        # Preprocess and predict
        processed_img = self.preprocess_image(image)
        batch_img = np.expand_dims(processed_img, axis=0)
        
        features = feature_model.predict(batch_img, verbose=0)[0]
        
        return features


class EnsemblePredictor:
    """Ensemble multiple models for improved predictions."""
    
    def __init__(
        self,
        model_paths: List[str],
        class_names: List[str],
        weights: List[float] = None
    ):
        """
        Initialize ensemble predictor.
        
        Args:
            model_paths: List of paths to trained models
            class_names: List of class names
            weights: Optional weights for each model
        """
        self.predictors = [
            DiseasePredictor(path, class_names)
            for path in model_paths
        ]
        
        if weights is None:
            weights = [1.0 / len(model_paths)] * len(model_paths)
        
        self.weights = np.array(weights)
        self.class_names = class_names
        
        logger.info(f"Initialized ensemble with {len(self.predictors)} models")
    
    def predict(self, image: Union[str, np.ndarray]) -> Dict[str, any]:
        """
        Predict using ensemble of models.
        
        Args:
            image: Image path or array
            
        Returns:
            Aggregated prediction results
        """
        # Get predictions from all models
        all_probs = []
        for predictor in self.predictors:
            result = predictor.predict(image, top_k=1)
            all_probs.append(result['all_probabilities'])
        
        # Weighted average
        ensemble_probs = np.average(all_probs, axis=0, weights=self.weights)
        
        # Get top prediction
        top_idx = np.argmax(ensemble_probs)
        
        return {
            'predicted_class': self.class_names[top_idx],
            'confidence': float(ensemble_probs[top_idx]),
            'all_probabilities': ensemble_probs.tolist()
        }
