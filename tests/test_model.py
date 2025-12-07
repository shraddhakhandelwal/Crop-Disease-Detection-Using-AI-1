"""
Unit tests for model architecture.
"""
import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import DiseaseClassifier, create_simple_cnn


class TestDiseaseClassifier:
    """Test cases for DiseaseClassifier."""
    
    def test_initialization(self):
        """Test classifier initialization."""
        classifier = DiseaseClassifier(
            model_name='efficientnet_b0',
            num_classes=10,
            input_shape=(224, 224, 3)
        )
        
        assert classifier.model_name == 'efficientnet_b0'
        assert classifier.num_classes == 10
        assert classifier.input_shape == (224, 224, 3)
    
    def test_invalid_model_name(self):
        """Test initialization with invalid model name."""
        with pytest.raises(ValueError):
            DiseaseClassifier(model_name='invalid_model')
    
    def test_build_model(self):
        """Test model building."""
        classifier = DiseaseClassifier(
            model_name='efficientnet_b0',
            num_classes=5
        )
        
        model = classifier.build_model()
        
        assert model is not None
        assert len(model.layers) > 0
        
        # Check output shape
        assert model.output_shape == (None, 5)
    
    def test_compile_model(self):
        """Test model compilation."""
        classifier = DiseaseClassifier(num_classes=5)
        model = classifier.build_model()
        
        compiled_model = DiseaseClassifier.compile_model(
            model,
            learning_rate=0.001,
            optimizer='adam'
        )
        
        assert compiled_model.optimizer is not None
    
    def test_model_prediction(self):
        """Test model can make predictions."""
        classifier = DiseaseClassifier(num_classes=5)
        model = classifier.build_model()
        model = DiseaseClassifier.compile_model(model)
        
        # Create dummy input
        dummy_input = np.random.rand(1, 224, 224, 3)
        
        # Make prediction
        prediction = model.predict(dummy_input, verbose=0)
        
        assert prediction.shape == (1, 5)
        assert np.allclose(prediction.sum(), 1.0, atol=1e-5)  # Softmax outputs sum to 1


class TestSimpleCNN:
    """Test cases for simple CNN."""
    
    def test_create_simple_cnn(self):
        """Test creating simple CNN."""
        model = create_simple_cnn(num_classes=10)
        
        assert model is not None
        assert model.output_shape == (None, 10)
    
    def test_simple_cnn_prediction(self):
        """Test simple CNN can make predictions."""
        model = create_simple_cnn(num_classes=3)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        dummy_input = np.random.rand(2, 224, 224, 3)
        prediction = model.predict(dummy_input, verbose=0)
        
        assert prediction.shape == (2, 3)
