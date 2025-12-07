"""
Unit tests for data preprocessing module.
"""
import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import ImageDataLoader, DataAugmentor


class TestImageDataLoader:
    """Test cases for ImageDataLoader."""
    
    def test_initialization(self, tmp_path):
        """Test loader initialization."""
        loader = ImageDataLoader(str(tmp_path), image_size=(224, 224))
        assert loader.image_size == (224, 224)
        assert loader.data_dir == tmp_path
    
    def test_load_image_invalid_path(self):
        """Test loading invalid image path."""
        loader = ImageDataLoader("fake_path")
        with pytest.raises(ValueError):
            loader.load_image("nonexistent.jpg")
    
    def test_split_data(self):
        """Test data splitting."""
        loader = ImageDataLoader("fake_path")
        
        # Create dummy data
        images = np.random.rand(100, 224, 224, 3)
        labels = np.random.randint(0, 10, 100)
        
        splits = loader.split_data(images, labels, val_split=0.2, test_split=0.1)
        
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        
        X_train, y_train = splits['train']
        X_val, y_val = splits['val']
        X_test, y_test = splits['test']
        
        # Check sizes
        assert len(X_train) + len(X_val) + len(X_test) == 100
        assert len(X_test) == 10  # 10% of 100
        assert len(X_val) == 18   # 20% of remaining 90


class TestDataAugmentor:
    """Test cases for DataAugmentor."""
    
    def test_initialization(self):
        """Test augmentor initialization."""
        config = {
            'rotation_range': 20,
            'horizontal_flip': True,
            'brightness_range': [0.8, 1.2]
        }
        
        augmentor = DataAugmentor(config)
        assert augmentor.config == config
        assert augmentor.train_transform is not None
        assert augmentor.val_transform is not None
    
    def test_augment_train(self):
        """Test training augmentation."""
        config = {'rotation_range': 20, 'horizontal_flip': True}
        augmentor = DataAugmentor(config)
        
        # Create dummy image
        image = np.random.rand(224, 224, 3).astype(np.float32)
        
        augmented = augmentor.augment_train(image)
        
        # Check output shape
        assert augmented.shape == image.shape
    
    def test_augment_val(self):
        """Test validation augmentation."""
        config = {}
        augmentor = DataAugmentor(config)
        
        image = np.random.rand(224, 224, 3).astype(np.float32)
        augmented = augmentor.augment_val(image)
        
        assert augmented.shape == image.shape
