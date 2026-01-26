"""
Visualization utilities for Credit Card Fraud Detection.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix


def plot_class_distribution(y, title: str = "Class Distribution", figsize=(10, 5)):
    """
    Plot the distribution of classes.

    Args:
        y: Target labels
        title: Plot title
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Count plot
    unique, counts = np.unique(y, return_counts=True)
    colors = ['#2ecc71', '#e74c3c']
    labels = ['Normal (0)', 'Fraud (1)']

    axes[0].bar(labels, counts, color=colors, edgecolor='black')
    axes[0].set_title(title)
    axes[0].set_ylabel('Count')
    for i, (count, label) in enumerate(zip(counts, labels)):
        axes[0].text(i, count + max(counts)*0.01, f'{count:,}', ha='center', fontsize=10)

    # Pie chart
    axes[1].pie(counts, labels=labels, autopct='%1.2f%%', colors=colors,
                explode=(0, 0.1), shadow=True, startangle=90)
    axes[1].set_title('Class Percentage')

    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix", figsize=(8, 6)):
    """
    Plot confusion matrix heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)

    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / total * 100
            current_text = ax.texts[i * 2 + j]
            current_text.set_text(f'{cm[i, j]}\n({percentage:.2f}%)')

    plt.tight_layout()
    return fig


def plot_roc_curves(models_results: dict, y_test, figsize=(10, 8)):
    """
    Plot ROC curves for multiple models.

    Args:
        models_results: Dictionary with model results containing y_prob
        y_test: True test labels
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set1(np.linspace(0, 1, len(models_results)))

    for (name, results), color in zip(models_results.items(), colors):
        y_prob = results['y_prob']
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = results['roc_auc']
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_precision_recall_curves(models_results: dict, y_test, figsize=(10, 8)):
    """
    Plot Precision-Recall curves for multiple models.

    Args:
        models_results: Dictionary with model results containing y_prob
        y_test: True test labels
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set1(np.linspace(0, 1, len(models_results)))

    for (name, results), color in zip(models_results.items(), colors):
        y_prob = results['y_prob']
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        avg_precision = results['avg_precision']
        ax.plot(recall, precision, color=color, lw=2,
                label=f'{name} (AP = {avg_precision:.4f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves Comparison')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n: int = 20,
                           title: str = "Feature Importance", figsize=(10, 8)):
    """
    Plot feature importance for tree-based models.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to display
        title: Plot title
        figsize: Figure size
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return None

    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(indices))
    ax.barh(y_pos, importance[indices], color='steelblue', edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_metrics_comparison(models_results: dict, figsize=(12, 6)):
    """
    Plot bar chart comparing metrics across models.

    Args:
        models_results: Dictionary with model results
        figsize: Figure size
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    model_names = list(models_results.keys())

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)

    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

    for i, (name, results) in enumerate(models_results.items()):
        values = [results[m] for m in metrics]
        offset = (i - len(model_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name, color=colors[i], edgecolor='black')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)

    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_sampling_comparison(sampling_results: dict, figsize=(12, 5)):
    """
    Plot comparison of different sampling methods.

    Args:
        sampling_results: Dictionary with {method_name: (X, y)}
        figsize: Figure size
    """
    n_methods = len(sampling_results)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)

    if n_methods == 1:
        axes = [axes]

    colors = ['#2ecc71', '#e74c3c']

    for ax, (method, (X, y)) in zip(axes, sampling_results.items()):
        unique, counts = np.unique(y, return_counts=True)
        labels = ['Normal', 'Fraud']

        ax.bar(labels, counts, color=colors, edgecolor='black')
        ax.set_title(f'{method}\n(Total: {len(y):,})')
        ax.set_ylabel('Count')

        for i, count in enumerate(counts):
            ax.text(i, count + max(counts)*0.01, f'{count:,}', ha='center', fontsize=9)

    plt.suptitle('Sampling Methods Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig
