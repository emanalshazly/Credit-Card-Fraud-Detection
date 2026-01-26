"""
Handlers for imbalanced dataset - SMOTE, Undersampling, Class Weights.
"""
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.utils.class_weight import compute_class_weight


def apply_smote(X_train, y_train, random_state: int = 42, sampling_strategy: float = 0.5):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique).

    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed
        sampling_strategy: Ratio of minority to majority class after resampling

    Returns:
        X_resampled, y_resampled
    """
    print("Applying SMOTE...")
    print(f"Before SMOTE: {Counter(y_train)}")

    smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE: {Counter(y_resampled)}")
    return X_resampled, y_resampled


def apply_undersampling(X_train, y_train, random_state: int = 42, sampling_strategy: float = 0.5):
    """
    Apply Random Undersampling to the majority class.

    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed
        sampling_strategy: Ratio of minority to majority class after resampling

    Returns:
        X_resampled, y_resampled
    """
    print("Applying Random Undersampling...")
    print(f"Before Undersampling: {Counter(y_train)}")

    undersampler = RandomUnderSampler(random_state=random_state, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)

    print(f"After Undersampling: {Counter(y_resampled)}")
    return X_resampled, y_resampled


def apply_smote_tomek(X_train, y_train, random_state: int = 42):
    """
    Apply SMOTE combined with Tomek links for cleaning.

    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed

    Returns:
        X_resampled, y_resampled
    """
    print("Applying SMOTE + Tomek Links...")
    print(f"Before SMOTE-Tomek: {Counter(y_train)}")

    smote_tomek = SMOTETomek(random_state=random_state)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

    print(f"After SMOTE-Tomek: {Counter(y_resampled)}")
    return X_resampled, y_resampled


def get_class_weights(y_train, weight_type: str = 'balanced') -> dict:
    """
    Calculate class weights for handling imbalanced data.

    Args:
        y_train: Training labels
        weight_type: Type of weighting ('balanced' or 'custom')

    Returns:
        Dictionary mapping class labels to weights
    """
    classes = np.unique(y_train)

    if weight_type == 'balanced':
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
    else:
        # Custom weights based on inverse frequency
        counter = Counter(y_train)
        total = sum(counter.values())
        class_weights = {cls: total / count for cls, count in counter.items()}

    print(f"Class weights: {class_weights}")
    return class_weights


def compare_sampling_methods(X_train, y_train, random_state: int = 42):
    """
    Compare different sampling methods and return results.

    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed

    Returns:
        Dictionary with resampled datasets
    """
    results = {
        'original': (X_train, y_train),
        'smote': apply_smote(X_train, y_train, random_state),
        'undersampling': apply_undersampling(X_train, y_train, random_state),
        'smote_tomek': apply_smote_tomek(X_train, y_train, random_state)
    }

    return results
