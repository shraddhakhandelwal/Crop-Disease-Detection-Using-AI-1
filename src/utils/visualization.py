"""
Visualization utilities for model analysis and results.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
import json
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_training_history(
    history_path: str,
    save_path: str = None,
    metrics: List[str] = None
) -> None:
    """
    Plot training history metrics.
    
    Args:
        history_path: Path to saved history JSON file
        save_path: Path to save plot
        metrics: List of metrics to plot
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    if metrics is None:
        metrics = ['accuracy', 'loss']
    
    # Create subplots
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Plot training metric
        if metric in history:
            ax.plot(history[metric], label=f'Train {metric}', linewidth=2)
        
        # Plot validation metric
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(history[val_metric], label=f'Val {metric}', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'Model {metric.capitalize()}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training history plot to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (16, 14)
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save plot
        normalize: Whether to normalize values
        figsize: Figure size
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True if len(class_names) <= 20 else False,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    
    plt.show()


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str = None
) -> Dict:
    """
    Generate and save classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save report
        
    Returns:
        Classification report as dictionary
    """
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True
    )
    
    # Print report
    print("\nClassification Report:")
    print("=" * 80)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Save to JSON
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved classification report to {save_path}")
    
    return report


def plot_class_distribution(
    labels: np.ndarray,
    class_names: List[str],
    save_path: str = None,
    title: str = "Class Distribution"
) -> None:
    """
    Plot distribution of classes.
    
    Args:
        labels: Array of labels
        class_names: List of class names
        save_path: Path to save plot
        title: Plot title
    """
    # Count occurrences
    unique, counts = np.unique(labels, return_counts=True)
    
    # Sort by count
    sorted_indices = np.argsort(counts)[::-1]
    sorted_names = [class_names[i] for i in unique[sorted_indices]]
    sorted_counts = counts[sorted_indices]
    
    # Plot
    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(sorted_counts)), sorted_counts, color='steelblue', alpha=0.8)
    
    # Color bars based on frequency
    max_count = max(counts)
    for i, bar in enumerate(bars):
        ratio = sorted_counts[i] / max_count
        bar.set_color(plt.cm.viridis(ratio))
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=90, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, count in enumerate(sorted_counts):
        plt.text(i, count, str(count), ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved class distribution plot to {save_path}")
    
    plt.show()


def plot_prediction_confidence(
    predictions: List[Dict],
    save_path: str = None
) -> None:
    """
    Plot distribution of prediction confidences.
    
    Args:
        predictions: List of prediction dictionaries
        save_path: Path to save plot
    """
    confidences = [p['confidence'] for p in predictions]
    
    plt.figure(figsize=(10, 6))
    
    # Histogram
    plt.hist(confidences, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    
    # Add mean line
    mean_conf = np.mean(confidences)
    plt.axvline(mean_conf, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_conf:.3f}')
    
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confidence distribution plot to {save_path}")
    
    plt.show()


def plot_roc_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: List[str],
    save_path: str = None,
    plot_micro_macro: bool = True
) -> None:
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred_proba: Predicted probabilities
        class_names: List of class names
        save_path: Path to save plot
        plot_micro_macro: Whether to plot micro/macro averages
    """
    n_classes = len(class_names)
    
    # Compute ROC curve for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Plot top 10 classes with best AUC
    top_classes = sorted(range(n_classes), key=lambda i: roc_auc[i], reverse=True)[:10]
    
    for i in top_classes:
        plt.plot(fpr[i], tpr[i], lw=2, alpha=0.7,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves (Top 10 Classes)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curves to {save_path}")
    
    plt.show()


def plot_sample_predictions(
    images: np.ndarray,
    true_labels: List[str],
    pred_labels: List[str],
    confidences: List[float],
    save_path: str = None,
    n_samples: int = 16
) -> None:
    """
    Plot grid of sample predictions.
    
    Args:
        images: Array of images
        true_labels: True class names
        pred_labels: Predicted class names
        confidences: Prediction confidences
        save_path: Path to save plot
        n_samples: Number of samples to plot
    """
    n_samples = min(n_samples, len(images))
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    for idx in range(n_samples):
        ax = axes[idx]
        
        # Display image
        img = images[idx]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        ax.imshow(img)
        ax.axis('off')
        
        # Set title with prediction info
        is_correct = true_labels[idx] == pred_labels[idx]
        color = 'green' if is_correct else 'red'
        
        title = f"True: {true_labels[idx]}\n"
        title += f"Pred: {pred_labels[idx]}\n"
        title += f"Conf: {confidences[idx]:.2%}"
        
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')
    
    # Hide empty subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved sample predictions to {save_path}")
    
    plt.show()
