"""
Model architectures for crop disease detection using transfer learning.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    EfficientNetB0, ResNet50, MobileNetV2, VGG16, InceptionV3
)
from typing import Tuple, Optional
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DiseaseClassifier:
    """Build and configure disease classification models."""
    
    MODEL_REGISTRY = {
        'efficientnet_b0': EfficientNetB0,
        'resnet50': ResNet50,
        'mobilenet_v2': MobileNetV2,
        'vgg16': VGG16,
        'inceptionv3': InceptionV3,
    }
    
    def __init__(
        self,
        model_name: str = 'efficientnet_b0',
        num_classes: int = 38,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        """
        Initialize disease classifier.
        
        Args:
            model_name: Name of base model architecture
            num_classes: Number of disease classes
            input_shape: Input image shape (H, W, C)
            pretrained: Use ImageNet pretrained weights
            dropout_rate: Dropout rate for regularization
        """
        self.model_name = model_name.lower()
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        
        if self.model_name not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(self.MODEL_REGISTRY.keys())}"
            )
        
        logger.info(f"Initializing {model_name} with {num_classes} classes")
    
    def build_model(self, fine_tune_layers: Optional[int] = None) -> Model:
        """
        Build complete model with transfer learning.
        
        Args:
            fine_tune_layers: Number of layers to unfreeze for fine-tuning
            
        Returns:
            Compiled Keras model
        """
        # Get base model
        base_model_class = self.MODEL_REGISTRY[self.model_name]
        weights = 'imagenet' if self.pretrained else None
        
        base_model = base_model_class(
            include_top=False,
            weights=weights,
            input_shape=self.input_shape,
            pooling='avg'
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Build classification head
        inputs = keras.Input(shape=self.input_shape)
        
        # Data augmentation can be added here if needed
        x = inputs
        
        # Base model
        x = base_model(x, training=False)
        
        # Classification layers
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(512, activation='relu', name='fc1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(256, activation='relu', name='fc2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate / 2)(x)
        
        # Output layer
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='predictions'
        )(x)
        
        model = Model(inputs=inputs, outputs=outputs, name=f'{self.model_name}_classifier')
        
        # Optionally unfreeze top layers for fine-tuning
        if fine_tune_layers and fine_tune_layers > 0:
            self._unfreeze_layers(base_model, fine_tune_layers)
        
        logger.info(f"Built model with {model.count_params():,} total parameters")
        logger.info(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
        
        return model
    
    def _unfreeze_layers(self, base_model: Model, num_layers: int) -> None:
        """
        Unfreeze top N layers of base model for fine-tuning.
        
        Args:
            base_model: Base model to modify
            num_layers: Number of layers to unfreeze
        """
        base_model.trainable = True
        
        # Freeze all layers except the last num_layers
        for layer in base_model.layers[:-num_layers]:
            layer.trainable = False
        
        logger.info(f"Unfroze top {num_layers} layers for fine-tuning")
    
    @staticmethod
    def compile_model(
        model: Model,
        learning_rate: float = 0.001,
        optimizer: str = 'adam',
        loss: str = 'categorical_crossentropy',
        metrics: list = None
    ) -> Model:
        """
        Compile model with optimizer and loss.
        
        Args:
            model: Model to compile
            learning_rate: Learning rate
            optimizer: Optimizer name
            loss: Loss function
            metrics: List of metrics
            
        Returns:
            Compiled model
        """
        if metrics is None:
            metrics = ['accuracy', 
                      keras.metrics.Precision(name='precision'),
                      keras.metrics.Recall(name='recall'),
                      keras.metrics.AUC(name='auc')]
        
        # Select optimizer
        if optimizer.lower() == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer.lower() == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Compiled model with {optimizer} optimizer (lr={learning_rate})")
        
        return model
    
    @staticmethod
    def get_model_summary(model: Model) -> str:
        """
        Get formatted model summary.
        
        Args:
            model: Model to summarize
            
        Returns:
            String representation of model
        """
        from io import StringIO
        import sys
        
        stream = StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_string = stream.getvalue()
        stream.close()
        
        return summary_string


def create_simple_cnn(
    num_classes: int,
    input_shape: Tuple[int, int, int] = (224, 224, 3)
) -> Model:
    """
    Create a simple CNN from scratch (for comparison/baseline).
    
    Args:
        num_classes: Number of output classes
        input_shape: Input image shape
        
    Returns:
        Keras model
    """
    inputs = keras.Input(shape=input_shape)
    
    # Conv Block 1
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(x)
    
    # Conv Block 2
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(x)
    
    # Conv Block 3
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Conv Block 4
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Classifier
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='simple_cnn')
    
    logger.info(f"Created simple CNN with {model.count_params():,} parameters")
    
    return model
