"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for model interpretability.
"""
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Union
from pathlib import Path
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class GradCAM:
    """Generate Grad-CAM heatmaps for CNN predictions."""
    
    def __init__(self, model: keras.Model, layer_name: str = None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Trained Keras model
            layer_name: Name of convolutional layer to visualize
        """
        self.model = model
        
        # Auto-detect last conv layer if not specified
        if layer_name is None:
            layer_name = self._find_last_conv_layer()
        
        self.layer_name = layer_name
        logger.info(f"Using layer '{layer_name}' for Grad-CAM")
        
        # Create gradient model
        self.grad_model = self._build_grad_model()
    
    def _find_last_conv_layer(self) -> str:
        """Find the last convolutional layer in the model."""
        for layer in reversed(self.model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                return layer.name
            # Check if it's a Model (e.g., base model in transfer learning)
            if isinstance(layer, keras.Model):
                for sublayer in reversed(layer.layers):
                    if isinstance(sublayer, keras.layers.Conv2D):
                        return sublayer.name
        
        raise ValueError("No convolutional layer found in model")
    
    def _build_grad_model(self) -> keras.Model:
        """Build model that returns gradients and activations."""
        # Find the target layer
        target_layer = None
        for layer in self.model.layers:
            if layer.name == self.layer_name:
                target_layer = layer
                break
            # Check nested models
            if isinstance(layer, keras.Model):
                for sublayer in layer.layers:
                    if sublayer.name == self.layer_name:
                        target_layer = sublayer
                        break
        
        if target_layer is None:
            raise ValueError(f"Layer '{self.layer_name}' not found in model")
        
        return keras.Model(
            inputs=self.model.input,
            outputs=[target_layer.output, self.model.output]
        )
    
    def compute_heatmap(
        self,
        image: np.ndarray,
        class_idx: int = None,
        eps: float = 1e-8
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.
        
        Args:
            image: Preprocessed input image (with batch dimension)
            class_idx: Target class index (None for predicted class)
            eps: Small epsilon for numerical stability
            
        Returns:
            Heatmap as numpy array
        """
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Get activations and predictions
            conv_outputs, predictions = self.grad_model(image)
            
            # Use predicted class if not specified
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            # Get the score for the target class
            class_channel = predictions[:, class_idx]
        
        # Compute gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the feature maps by gradient importance
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        # Multiply each feature map by its importance
        for i in range(pooled_grads.shape[0]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Average over all feature maps
        heatmap = np.mean(conv_outputs, axis=-1)
        
        # ReLU on heatmap (only positive influence)
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize to [0, 1]
        if np.max(heatmap) > eps:
            heatmap = heatmap / np.max(heatmap)
        
        return heatmap
    
    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        original_image: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            heatmap: Grad-CAM heatmap
            original_image: Original image (RGB)
            alpha: Transparency of heatmap
            colormap: OpenCV colormap to use
            
        Returns:
            Overlayed image
        """
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure original image is uint8
        if original_image.dtype != np.uint8:
            original_image = np.uint8(255 * original_image)
        
        # Overlay
        overlayed = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        
        return overlayed
    
    def generate_gradcam(
        self,
        image: np.ndarray,
        original_image: np.ndarray,
        class_idx: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate complete Grad-CAM visualization.
        
        Args:
            image: Preprocessed image with batch dimension
            original_image: Original image for overlay
            class_idx: Target class (None for predicted)
            
        Returns:
            Tuple of (heatmap, overlayed_image)
        """
        # Compute heatmap
        heatmap = self.compute_heatmap(image, class_idx)
        
        # Overlay on original
        overlayed = self.overlay_heatmap(heatmap, original_image)
        
        return heatmap, overlayed


def visualize_gradcam(
    model: keras.Model,
    image_path: str,
    save_path: str = None,
    class_idx: int = None,
    image_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Convenience function to generate and save Grad-CAM visualization.
    
    Args:
        model: Trained model
        image_path: Path to input image
        save_path: Path to save visualization
        class_idx: Target class index
        image_size: Model input size
        
    Returns:
        Overlayed Grad-CAM image
    """
    # Load and preprocess image
    original = cv2.imread(str(image_path))
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Prepare for model
    img = cv2.resize(original, image_size)
    img_normalized = img.astype(np.float32) / 255.0
    
    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_normalized - mean) / std
    
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Generate Grad-CAM
    gradcam = GradCAM(model)
    heatmap, overlayed = gradcam.generate_gradcam(img_batch, img, class_idx)
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        overlayed_bgr = cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), overlayed_bgr)
        logger.info(f"Saved Grad-CAM visualization to {save_path}")
    
    return overlayed
