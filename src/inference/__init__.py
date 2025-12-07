"""
Inference module initialization.
"""
from .predictor import DiseasePredictor, EnsemblePredictor
from .gradcam import GradCAM, visualize_gradcam

__all__ = [
    'DiseasePredictor',
    'EnsemblePredictor',
    'GradCAM',
    'visualize_gradcam',
]
