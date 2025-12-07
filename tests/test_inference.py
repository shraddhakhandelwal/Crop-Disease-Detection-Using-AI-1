"""
Unit tests for inference module.
"""
import pytest
import numpy as np
from pathlib import Path
import sys
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import DiseasePredictor


class TestDiseasePredictor:
    """Test cases for DiseasePredictor."""
    
    @pytest.fixture
    def mock_model(self, tmp_path):
        """Create a mock model file."""
        model_path = tmp_path / "test_model.h5"
        model_path.touch()
        return model_path
    
    @pytest.fixture
    def class_names(self):
        """Sample class names."""
        return ['Disease_A', 'Disease_B', 'Disease_C']
    
    def test_preprocess_image(self, mock_model, class_names):
        """Test image preprocessing."""
        with patch('tensorflow.keras.models.load_model'):
            predictor = DiseasePredictor(
                model_path=str(mock_model),
                class_names=class_names
            )
            
            # Create dummy image
            image = np.random.rand(300, 300, 3).astype(np.uint8)
            
            processed = predictor.preprocess_image(image)
            
            # Check output
            assert processed.shape == (224, 224, 3)
            assert processed.dtype == np.float32
    
    def test_predict_format(self, mock_model, class_names):
        """Test prediction output format."""
        with patch('tensorflow.keras.models.load_model') as mock_load:
            # Create mock model
            mock_keras_model = Mock()
            mock_keras_model.predict.return_value = np.array([[0.7, 0.2, 0.1]])
            mock_load.return_value = mock_keras_model
            
            predictor = DiseasePredictor(
                model_path=str(mock_model),
                class_names=class_names
            )
            
            # Create dummy image
            image = np.random.rand(224, 224, 3).astype(np.uint8)
            
            result = predictor.predict(image, top_k=2)
            
            # Check result structure
            assert 'predicted_class' in result
            assert 'confidence' in result
            assert 'is_confident' in result
            assert 'top_predictions' in result
            assert len(result['top_predictions']) == 2
