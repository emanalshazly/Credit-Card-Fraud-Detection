"""
Handlers for imbalanced dataset - Comprehensive sampling strategies.

Methods:
1. SMOTE - Synthetic Minority Over-sampling
2. ADASYN - Adaptive Synthetic Sampling
3. Borderline-SMOTE - Focus on decision boundary
4. SMOTE-Tomek - Hybrid cleaning
5. SMOTE-ENN - Hybrid with Edited Nearest Neighbors
6. Random Undersampling
7. NearMiss - Informed undersampling
8. Cluster Centroids - Centroid-based undersampling
9. Class Weights - Cost-sensitive learning
"""
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss, ClusterCentroids
from imblearn.combine import SMOTETomek, SMOTEENN
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


def apply_adasyn(X_train, y_train, random_state: int = 42, sampling_strategy: float = 0.5):
    """
    Apply ADASYN (Adaptive Synthetic Sampling).

    ADASYN focuses on generating samples in regions where the class
    imbalance is more severe, adapting to local density.

    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed
        sampling_strategy: Target ratio

    Returns:
        X_resampled, y_resampled
    """
    print("Applying ADASYN...")
    print(f"Before ADASYN: {Counter(y_train)}")

    adasyn = ADASYN(random_state=random_state, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

    print(f"After ADASYN: {Counter(y_resampled)}")
    return X_resampled, y_resampled


def apply_borderline_smote(X_train, y_train, random_state: int = 42,
                           sampling_strategy: float = 0.5, kind: str = 'borderline-1'):
    """
    Apply Borderline-SMOTE.

    Focuses on generating synthetic samples along the decision boundary
    where misclassification is more likely.

    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed
        sampling_strategy: Target ratio
        kind: 'borderline-1' or 'borderline-2'

    Returns:
        X_resampled, y_resampled
    """
    print(f"Applying Borderline-SMOTE ({kind})...")
    print(f"Before Borderline-SMOTE: {Counter(y_train)}")

    borderline = BorderlineSMOTE(random_state=random_state,
                                  sampling_strategy=sampling_strategy,
                                  kind=kind)
    X_resampled, y_resampled = borderline.fit_resample(X_train, y_train)

    print(f"After Borderline-SMOTE: {Counter(y_resampled)}")
    return X_resampled, y_resampled


def apply_smote_enn(X_train, y_train, random_state: int = 42):
    """
    Apply SMOTE combined with Edited Nearest Neighbors (ENN) cleaning.

    ENN removes samples whose class differs from the majority of their
    k-nearest neighbors, cleaning noisy samples after SMOTE.

    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed

    Returns:
        X_resampled, y_resampled
    """
    print("Applying SMOTE + ENN...")
    print(f"Before SMOTE-ENN: {Counter(y_train)}")

    smote_enn = SMOTEENN(random_state=random_state)
    X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)

    print(f"After SMOTE-ENN: {Counter(y_resampled)}")
    return X_resampled, y_resampled


def apply_nearmiss(X_train, y_train, version: int = 1, n_neighbors: int = 3):
    """
    Apply NearMiss undersampling.

    Versions:
    - NearMiss-1: Selects majority samples with smallest avg distance to k nearest minority
    - NearMiss-2: Selects majority samples with smallest avg distance to k farthest minority
    - NearMiss-3: Keeps majority samples that are farthest from their nearest minority

    Args:
        X_train: Training features
        y_train: Training labels
        version: NearMiss version (1, 2, or 3)
        n_neighbors: Number of neighbors

    Returns:
        X_resampled, y_resampled
    """
    print(f"Applying NearMiss-{version}...")
    print(f"Before NearMiss: {Counter(y_train)}")

    nearmiss = NearMiss(version=version, n_neighbors=n_neighbors)
    X_resampled, y_resampled = nearmiss.fit_resample(X_train, y_train)

    print(f"After NearMiss: {Counter(y_resampled)}")
    return X_resampled, y_resampled


def apply_cluster_centroids(X_train, y_train, random_state: int = 42):
    """
    Apply Cluster Centroids undersampling.

    Uses K-Means to find centroids of clusters in the majority class,
    then replaces majority samples with centroids.

    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed

    Returns:
        X_resampled, y_resampled
    """
    print("Applying Cluster Centroids...")
    print(f"Before Cluster Centroids: {Counter(y_train)}")

    cc = ClusterCentroids(random_state=random_state)
    X_resampled, y_resampled = cc.fit_resample(X_train, y_train)

    print(f"After Cluster Centroids: {Counter(y_resampled)}")
    return X_resampled, y_resampled


def compare_all_sampling_methods(X_train, y_train, random_state: int = 42):
    """
    Compare all available sampling methods.

    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed

    Returns:
        Dictionary with all resampled datasets
    """
    print("\n" + "="*60)
    print("COMPARING ALL SAMPLING METHODS")
    print("="*60)

    results = {
        'original': (X_train, y_train),
    }

    # Oversampling methods
    try:
        results['smote'] = apply_smote(X_train, y_train, random_state)
    except Exception as e:
        print(f"SMOTE failed: {e}")

    try:
        results['adasyn'] = apply_adasyn(X_train, y_train, random_state)
    except Exception as e:
        print(f"ADASYN failed: {e}")

    try:
        results['borderline_smote'] = apply_borderline_smote(X_train, y_train, random_state)
    except Exception as e:
        print(f"Borderline-SMOTE failed: {e}")

    # Hybrid methods
    try:
        results['smote_tomek'] = apply_smote_tomek(X_train, y_train, random_state)
    except Exception as e:
        print(f"SMOTE-Tomek failed: {e}")

    try:
        results['smote_enn'] = apply_smote_enn(X_train, y_train, random_state)
    except Exception as e:
        print(f"SMOTE-ENN failed: {e}")

    # Undersampling methods
    try:
        results['undersampling'] = apply_undersampling(X_train, y_train, random_state)
    except Exception as e:
        print(f"Undersampling failed: {e}")

    try:
        results['nearmiss'] = apply_nearmiss(X_train, y_train)
    except Exception as e:
        print(f"NearMiss failed: {e}")

    # Summary
    print("\n" + "="*60)
    print("SAMPLING COMPARISON SUMMARY")
    print("="*60)
    for method, (X, y) in results.items():
        fraud_count = sum(y == 1)
        normal_count = sum(y == 0)
        ratio = fraud_count / normal_count if normal_count > 0 else 0
        print(f"{method:20s}: Normal={normal_count:,}, Fraud={fraud_count:,}, Ratio={ratio:.2%}")

    return results


def get_best_sampling_method(X_train, y_train, X_test, y_test,
                            model_class, random_state: int = 42) -> str:
    """
    Find best sampling method for a given model by comparing F1 scores.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model_class: Model class to use (e.g., RandomForestClassifier)
        random_state: Random seed

    Returns:
        Name of best sampling method
    """
    from sklearn.metrics import f1_score

    results = compare_all_sampling_methods(X_train, y_train, random_state)

    best_method = 'original'
    best_f1 = 0

    print("\n" + "="*60)
    print("FINDING BEST SAMPLING METHOD")
    print("="*60)

    for method, (X_resampled, y_resampled) in results.items():
        try:
            model = model_class(random_state=random_state, n_jobs=-1)
            if hasattr(model, 'class_weight'):
                model.set_params(class_weight='balanced')

            model.fit(X_resampled, y_resampled)
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred)

            print(f"{method:20s}: F1 = {f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_method = method
        except Exception as e:
            print(f"{method:20s}: Failed - {e}")

    print(f"\n✅ Best method: {best_method} (F1 = {best_f1:.4f})")
    return best_method
