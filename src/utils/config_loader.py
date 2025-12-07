"""
Utility functions for loading and parsing configuration files.
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract data configuration."""
    return config.get('data', {})


def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model configuration."""
    return config.get('model', {})


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract training configuration."""
    return config.get('training', {})


def get_augmentation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract augmentation configuration."""
    return config.get('augmentation', {})


def create_directory_structure(config: Dict[str, Any]) -> None:
    """
    Create necessary directories based on configuration.
    
    Args:
        config: Configuration dictionary
    """
    paths = config.get('paths', {})
    
    directories = [
        paths.get('model_dir', 'models'),
        paths.get('checkpoint_dir', 'models/checkpoints'),
        paths.get('saved_model_dir', 'models/saved_models'),
        paths.get('logs_dir', 'logs'),
        paths.get('results_dir', 'results'),
        config['data']['raw_dir'],
        config['data']['processed_dir'],
        config['data']['samples_dir'],
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created/verified directory: {directory}")
