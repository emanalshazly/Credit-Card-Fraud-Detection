"""
Model training and evaluation for Credit Card Fraud Detection.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
import joblib


def train_logistic_regression(X_train, y_train, class_weight=None, random_state: int = 42):
    """
    Train a Logistic Regression model.

    Args:
        X_train: Training features
        y_train: Training labels
        class_weight: Class weights for handling imbalance
        random_state: Random seed

    Returns:
        Trained model
    """
    model = LogisticRegression(
        class_weight=class_weight,
        random_state=random_state,
        max_iter=1000,
        solver='lbfgs',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, class_weight=None, random_state: int = 42,
                        n_estimators: int = 100, max_depth: int = 10):
    """
    Train a Random Forest model.

    Args:
        X_train: Training features
        y_train: Training labels
        class_weight: Class weights for handling imbalance
        random_state: Random seed
        n_estimators: Number of trees
        max_depth: Maximum depth of trees

    Returns:
        Trained model
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, scale_pos_weight: float = None, random_state: int = 42,
                  n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1):
    """
    Train an XGBoost model.

    Args:
        X_train: Training features
        y_train: Training labels
        scale_pos_weight: Weight for positive class (fraud)
        random_state: Random seed
        n_estimators: Number of boosting rounds
        max_depth: Maximum depth of trees
        learning_rate: Learning rate

    Returns:
        Trained model
    """
    # Calculate scale_pos_weight if not provided
    if scale_pos_weight is None:
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name: str = "Model"):
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for display

    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'avg_precision': average_precision_score(y_test, y_prob),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred,
        'y_prob': y_prob
    }

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall (TPR):      {metrics['recall']:.4f}")
    print(f"F1-Score:          {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"Avg Precision:     {metrics['avg_precision']:.4f}")
    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))

    return metrics


def compare_models(models_dict: dict, X_test, y_test) -> dict:
    """
    Compare multiple models on the same test set.

    Args:
        models_dict: Dictionary of {model_name: model}
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary with metrics for all models
    """
    results = {}
    for name, model in models_dict.items():
        results[name] = evaluate_model(model, X_test, y_test, name)
    return results


def save_model(model, filepath: str):
    """Save model to disk."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str):
    """Load model from disk."""
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model
