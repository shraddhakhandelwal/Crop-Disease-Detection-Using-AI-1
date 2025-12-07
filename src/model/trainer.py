"""
Training utilities including callbacks and training loops.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
from datetime import datetime
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TrainingManager:
    """Manage model training process."""
    
    def __init__(self, config: Dict[str, Any], experiment_name: str = None):
        """
        Initialize training manager.
        
        Args:
            config: Training configuration dictionary
            experiment_name: Name for this training run
        """
        self.config = config
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        self.checkpoint_dir = Path(config['paths']['checkpoint_dir']) / self.experiment_name
        self.logs_dir = Path(config['paths']['logs_dir']) / self.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Training experiment: {self.experiment_name}")
    
    def get_callbacks(self) -> List[keras.callbacks.Callback]:
        """
        Create training callbacks.
        
        Returns:
            List of Keras callbacks
        """
        training_config = self.config['training']
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = self.checkpoint_dir / 'best_model.h5'
        checkpoint = ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=training_config.get('checkpoint_monitor', 'val_accuracy'),
            mode=training_config.get('checkpoint_mode', 'max'),
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor=training_config.get('checkpoint_monitor', 'val_accuracy'),
            patience=training_config.get('early_stopping_patience', 10),
            mode=training_config.get('checkpoint_mode', 'max'),
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        
        # Reduce learning rate
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=training_config.get('reduce_lr_patience', 5),
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard
        tensorboard = TensorBoard(
            log_dir=str(self.logs_dir),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard)
        
        # CSV Logger
        csv_logger = CSVLogger(
            filename=str(self.logs_dir / 'training_log.csv'),
            separator=',',
            append=False
        )
        callbacks.append(csv_logger)
        
        logger.info(f"Created {len(callbacks)} callbacks")
        
        return callbacks
    
    def compute_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights for imbalanced datasets.
        
        Args:
            y_train: Training labels (one-hot or categorical)
            
        Returns:
            Dictionary mapping class index to weight
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        # Convert one-hot to categorical if needed
        if len(y_train.shape) > 1:
            y_train = np.argmax(y_train, axis=1)
        
        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        logger.info(f"Computed class weights for {len(classes)} classes")
        logger.info(f"Weight range: {min(class_weights):.3f} - {max(class_weights):.3f}")
        
        return class_weight_dict
    
    def train_model(
        self,
        model: keras.Model,
        train_data: tuple,
        val_data: tuple,
        class_weights: Optional[Dict[int, float]] = None
    ) -> keras.callbacks.History:
        """
        Train the model.
        
        Args:
            model: Compiled Keras model
            train_data: Tuple of (X_train, y_train)
            val_data: Tuple of (X_val, y_val)
            class_weights: Optional class weights dictionary
            
        Returns:
            Training history
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        training_config = self.config['training']
        
        logger.info("Starting training...")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Epochs: {training_config['epochs']}")
        logger.info(f"Batch size: {self.config['data']['batch_size']}")
        
        history = model.fit(
            X_train, y_train,
            batch_size=self.config['data']['batch_size'],
            epochs=training_config['epochs'],
            validation_data=(X_val, y_val),
            callbacks=self.get_callbacks(),
            class_weight=class_weights,
            verbose=1
        )
        
        logger.info("Training completed!")
        
        return history
    
    def save_final_model(self, model: keras.Model, filename: str = 'final_model.h5') -> str:
        """
        Save final trained model.
        
        Args:
            model: Trained model
            filename: Output filename
            
        Returns:
            Path to saved model
        """
        save_dir = Path(self.config['paths']['saved_model_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / filename
        model.save(str(save_path))
        
        logger.info(f"Saved final model to {save_path}")
        
        return str(save_path)
    
    def save_training_history(self, history: keras.callbacks.History) -> str:
        """
        Save training history to file.
        
        Args:
            history: Training history object
            
        Returns:
            Path to saved history
        """
        import json
        
        history_path = self.logs_dir / 'history.json'
        
        # Convert history to serializable format
        history_dict = {
            key: [float(val) for val in values]
            for key, values in history.history.items()
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        logger.info(f"Saved training history to {history_path}")
        
        return str(history_path)
