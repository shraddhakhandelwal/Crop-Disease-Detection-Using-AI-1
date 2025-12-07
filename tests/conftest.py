"""
Pytest configuration file.
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'data': {
            'image_size': [224, 224],
            'batch_size': 32,
            'validation_split': 0.2,
            'test_split': 0.1,
            'seed': 42
        },
        'model': {
            'name': 'efficientnet_b0',
            'num_classes': 10,
            'pretrained': True,
            'dropout_rate': 0.3
        },
        'training': {
            'epochs': 10,
            'learning_rate': 0.001,
            'optimizer': 'adam'
        }
    }


@pytest.fixture
def sample_image():
    """Sample image for testing."""
    import numpy as np
    return np.random.rand(224, 224, 3).astype(np.float32)
