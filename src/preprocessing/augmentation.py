"""
Data augmentation utilities using Albumentations.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Dict, Any
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataAugmentor:
    """Handle data augmentation for training."""
    
    def __init__(self, config: Dict[str, Any], image_size: tuple = (224, 224)):
        """
        Initialize augmentor with configuration.
        
        Args:
            config: Augmentation configuration dictionary
            image_size: Target image size
        """
        self.config = config
        self.image_size = image_size
        self.train_transform = self._build_train_transforms()
        self.val_transform = self._build_val_transforms()
    
    def _build_train_transforms(self) -> A.Compose:
        """Build training augmentation pipeline."""
        transforms = []
        
        # Geometric transformations
        if self.config.get('rotation_range', 0) > 0:
            transforms.append(
                A.Rotate(
                    limit=self.config['rotation_range'],
                    p=0.5
                )
            )
        
        if self.config.get('horizontal_flip', False):
            transforms.append(A.HorizontalFlip(p=0.5))
        
        if self.config.get('vertical_flip', False):
            transforms.append(A.VerticalFlip(p=0.5))
        
        # Shift transformations
        shift_limit = max(
            self.config.get('width_shift_range', 0),
            self.config.get('height_shift_range', 0)
        )
        if shift_limit > 0:
            transforms.append(
                A.ShiftScaleRotate(
                    shift_limit=shift_limit,
                    scale_limit=self.config.get('zoom_range', 0),
                    rotate_limit=0,
                    p=0.5
                )
            )
        
        # Color augmentations
        if self.config.get('brightness_range'):
            brightness_range = self.config['brightness_range']
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=(brightness_range[0] - 1, brightness_range[1] - 1),
                    contrast_limit=0.2,
                    p=0.5
                )
            )
        
        # Additional augmentations for robustness
        transforms.extend([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        ])
        
        # Normalization
        transforms.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        
        logger.info(f"Built training augmentation pipeline with {len(transforms)} transforms")
        return A.Compose(transforms)
    
    def _build_val_transforms(self) -> A.Compose:
        """Build validation/test augmentation pipeline (normalization only)."""
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def augment_train(self, image: np.ndarray) -> np.ndarray:
        """
        Apply training augmentation to image.
        
        Args:
            image: Input image (H, W, C) in range [0, 1]
            
        Returns:
            Augmented image
        """
        # Albumentations expects uint8 images
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        
        augmented = self.train_transform(image=image)
        return augmented['image']
    
    def augment_val(self, image: np.ndarray) -> np.ndarray:
        """
        Apply validation augmentation to image.
        
        Args:
            image: Input image (H, W, C) in range [0, 1]
            
        Returns:
            Normalized image
        """
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        
        augmented = self.val_transform(image=image)
        return augmented['image']


def get_tensorflow_augmentation(config: Dict[str, Any]):
    """
    Create TensorFlow/Keras data augmentation layer.
    
    Args:
        config: Augmentation configuration
        
    Returns:
        Sequential model with augmentation layers
    """
    from tensorflow.keras import layers, Sequential
    
    augmentation_layers = []
    
    if config.get('rotation_range', 0) > 0:
        augmentation_layers.append(
            layers.RandomRotation(config['rotation_range'] / 360.0)
        )
    
    if config.get('horizontal_flip', False):
        augmentation_layers.append(layers.RandomFlip("horizontal"))
    
    if config.get('vertical_flip', False):
        augmentation_layers.append(layers.RandomFlip("vertical"))
    
    if config.get('zoom_range', 0) > 0:
        zoom = config['zoom_range']
        augmentation_layers.append(
            layers.RandomZoom((-zoom, zoom))
        )
    
    shift_w = config.get('width_shift_range', 0)
    shift_h = config.get('height_shift_range', 0)
    if shift_w > 0 or shift_h > 0:
        augmentation_layers.append(
            layers.RandomTranslation(shift_h, shift_w)
        )
    
    if config.get('brightness_range'):
        brightness_delta = config['brightness_range'][1] - 1.0
        augmentation_layers.append(
            layers.RandomBrightness(brightness_delta)
        )
    
    return Sequential(augmentation_layers, name="data_augmentation")
