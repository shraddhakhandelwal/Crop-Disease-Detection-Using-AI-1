"""
Data loading and preprocessing utilities for crop disease detection.
"""
import os
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ImageDataLoader:
    """Load and preprocess images for training and inference."""
    
    def __init__(self, data_dir: str, image_size: Tuple[int, int] = (224, 224)):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing class subdirectories with images
            image_size: Target image size (height, width)
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.class_names = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        if self.data_dir.exists():
            self._build_class_mapping()
    
    def _build_class_mapping(self) -> None:
        """Build mapping between class names and indices."""
        self.class_names = sorted([
            d.name for d in self.data_dir.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        logger.info(f"Found {len(self.class_names)} classes: {self.class_names[:5]}...")
    
    def load_image(self, image_path: str, normalize: bool = True) -> np.ndarray:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to image file
            normalize: Whether to normalize pixel values to [0, 1]
            
        Returns:
            Preprocessed image array
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, self.image_size)
        
        # Normalize
        if normalize:
            image = image.astype(np.float32) / 255.0
        
        return image
    
    def load_dataset(
        self,
        max_samples_per_class: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load all images and labels from the data directory.
        
        Args:
            max_samples_per_class: Maximum samples to load per class (for testing)
            
        Returns:
            Tuple of (images, labels, file_paths)
        """
        images = []
        labels = []
        file_paths = []
        
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(class_dir.glob(ext))
            
            # Limit samples if specified
            if max_samples_per_class:
                image_files = image_files[:max_samples_per_class]
            
            logger.info(f"Loading {len(image_files)} images from class '{class_name}'")
            
            for img_path in image_files:
                try:
                    image = self.load_image(img_path)
                    images.append(image)
                    labels.append(class_idx)
                    file_paths.append(str(img_path))
                except Exception as e:
                    logger.warning(f"Failed to load {img_path}: {e}")
        
        return np.array(images), np.array(labels), file_paths
    
    def split_data(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        val_split: float = 0.2,
        test_split: float = 0.1,
        random_state: int = 42
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            images: Image array
            labels: Label array
            val_split: Validation set proportion
            test_split: Test set proportion
            random_state: Random seed
            
        Returns:
            Dictionary with train, val, test splits
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels,
            test_size=test_split,
            random_state=random_state,
            stratify=labels
        )
        
        # Second split: train vs val
        val_size_adjusted = val_split / (1 - test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
