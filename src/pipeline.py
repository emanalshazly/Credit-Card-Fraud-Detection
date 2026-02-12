"""
Production ML Pipeline for Credit Card Fraud Detection.

End-to-end pipeline with:
1. Preprocessing
2. Feature Engineering
3. Resampling
4. Model Training
5. Threshold Optimization
6. Model Serialization
"""
import numpy as np
import pandas as pd
import joblib
from typing import Any, Dict, Tuple, Optional
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .feature_engineering import FraudFeatureEngineer
from .threshold_optimization import ThresholdOptimizer


class FraudDetectionPipeline(BaseEstimator, ClassifierMixin):
    """
    Complete fraud detection pipeline.

    Combines:
    - Preprocessing (scaling)
    - Feature Engineering
    - Model Training
    - Threshold Optimization
    - Prediction with optimized threshold
    """

    def __init__(self, model: Any = None,
                 apply_feature_engineering: bool = True,
                 apply_scaling: bool = True,
                 threshold_method: str = 'f1',
                 threshold: float = None):
        """
        Initialize pipeline.

        Args:
            model: Base model (default: XGBoost)
            apply_feature_engineering: Whether to create engineered features
            apply_scaling: Whether to scale features
            threshold_method: How to optimize threshold ('f1', 'f2', 'cost', 'fixed')
            threshold: Fixed threshold (if threshold_method='fixed')
        """
        self.model = model
        self.apply_feature_engineering = apply_feature_engineering
        self.apply_scaling = apply_scaling
        self.threshold_method = threshold_method
        self.threshold = threshold

        # Will be fitted
        self.feature_engineer_ = None
        self.scaler_ = None
        self.optimal_threshold_ = 0.5
        self.feature_names_ = None
        self.fitted_ = False

    def fit(self, X, y):
        """
        Fit the complete pipeline.

        Args:
            X: Training features (DataFrame)
            y: Training labels

        Returns:
            self
        """
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        y = np.array(y)

        # 1. Feature Engineering
        if self.apply_feature_engineering:
            self.feature_engineer_ = FraudFeatureEngineer()
            X = self.feature_engineer_.fit_transform(X)

        self.feature_names_ = list(X.columns)

        # 2. Scaling
        if self.apply_scaling:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=self.feature_names_)

        # 3. Initialize model if not provided
        if self.model is None:
            from xgboost import XGBClassifier
            neg_count = np.sum(y == 0)
            pos_count = np.sum(y == 1)
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            )

        # 4. Train model
        self.model.fit(X, y)

        # 5. Optimize threshold
        if self.threshold_method != 'fixed':
            y_prob = self.model.predict_proba(X)[:, 1]
            optimizer = ThresholdOptimizer(y, y_prob)

            if self.threshold_method == 'f1':
                self.optimal_threshold_, _ = optimizer.optimize_f1()
            elif self.threshold_method == 'f2':
                self.optimal_threshold_, _ = optimizer.optimize_fbeta(beta=2.0)
            elif self.threshold_method == 'cost':
                self.optimal_threshold_, _ = optimizer.optimize_cost()
            elif self.threshold_method == 'youden':
                self.optimal_threshold_, _ = optimizer.optimize_youden_j()
            else:
                self.optimal_threshold_ = 0.5
        else:
            self.optimal_threshold_ = self.threshold or 0.5

        self.fitted_ = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Features to predict

        Returns:
            Array of probabilities [n_samples, 2]
        """
        if not self.fitted_:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        # Apply transformations
        if self.apply_feature_engineering and self.feature_engineer_ is not None:
            X = self.feature_engineer_.transform(X)

        if self.apply_scaling and self.scaler_ is not None:
            X = pd.DataFrame(
                self.scaler_.transform(X),
                columns=self.feature_names_
            )

        return self.model.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        """
        Predict with optimized threshold.

        Args:
            X: Features to predict

        Returns:
            Array of predictions (0/1)
        """
        y_prob = self.predict_proba(X)[:, 1]
        return (y_prob >= self.optimal_threshold_).astype(int)

    def evaluate(self, X, y) -> Dict[str, float]:
        """
        Evaluate pipeline on test data.

        Args:
            X: Test features
            y: Test labels

        Returns:
            Dictionary with metrics
        """
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)[:, 1]

        return {
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_prob),
            'threshold': self.optimal_threshold_
        }

    def save(self, filepath: str):
        """Save pipeline to disk."""
        joblib.dump(self, filepath)
        print(f"Pipeline saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'FraudDetectionPipeline':
        """Load pipeline from disk."""
        pipeline = joblib.load(filepath)
        print(f"Pipeline loaded from {filepath}")
        return pipeline

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance."""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            df = pd.DataFrame({
                'feature': self.feature_names_[:len(importance)],
                'importance': importance
            }).sort_values('importance', ascending=False)
            return df.head(top_n)
        return None


class ModelEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble of fraud detection models.

    Combines predictions using:
    - Soft voting (average probabilities)
    - Hard voting (majority vote)
    - Weighted average
    - Stacking (meta-learner)
    """

    def __init__(self, models: Dict[str, Any] = None,
                 voting: str = 'soft',
                 weights: Dict[str, float] = None):
        """
        Initialize ensemble.

        Args:
            models: Dictionary of {name: model}
            voting: 'soft' (average proba) or 'hard' (majority vote)
            weights: Dictionary of {name: weight} for weighted voting
        """
        self.models = models or {}
        self.voting = voting
        self.weights = weights
        self.fitted_ = False

    def fit(self, X, y):
        """Fit all models."""
        for name, model in self.models.items():
            print(f"Fitting {name}...")
            model.fit(X, y)

        self.fitted_ = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Get ensemble probabilities."""
        if not self.fitted_:
            raise ValueError("Ensemble not fitted")

        probas = []
        weights = []

        for name, model in self.models.items():
            prob = model.predict_proba(X)[:, 1]
            probas.append(prob)

            if self.weights:
                weights.append(self.weights.get(name, 1.0))
            else:
                weights.append(1.0)

        # Weighted average
        weights = np.array(weights) / sum(weights)
        ensemble_prob = np.average(probas, axis=0, weights=weights)

        # Return as [n_samples, 2] array
        return np.column_stack([1 - ensemble_prob, ensemble_prob])

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """Predict with ensemble."""
        if self.voting == 'soft':
            y_prob = self.predict_proba(X)[:, 1]
            return (y_prob >= threshold).astype(int)
        else:
            # Hard voting
            votes = []
            for model in self.models.values():
                votes.append(model.predict(X))
            votes = np.array(votes)
            return (votes.mean(axis=0) >= 0.5).astype(int)

    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """Add model to ensemble."""
        self.models[name] = model
        if self.weights is None:
            self.weights = {}
        self.weights[name] = weight


def create_default_pipeline(X_train, y_train,
                           threshold_method: str = 'f1') -> FraudDetectionPipeline:
    """
    Create and fit default fraud detection pipeline.

    Args:
        X_train: Training features
        y_train: Training labels
        threshold_method: Threshold optimization method

    Returns:
        Fitted pipeline
    """
    from xgboost import XGBClassifier

    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )

    pipeline = FraudDetectionPipeline(
        model=model,
        apply_feature_engineering=True,
        apply_scaling=True,
        threshold_method=threshold_method
    )

    pipeline.fit(X_train, y_train)
    return pipeline


def create_ensemble_pipeline(X_train, y_train) -> ModelEnsemble:
    """
    Create ensemble of multiple models.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        Fitted ensemble
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

    models = {
        'logistic': LogisticRegression(
            class_weight='balanced', max_iter=1000,
            random_state=42, n_jobs=-1
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=200, max_depth=15,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'xgboost': XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42, use_label_encoder=False,
            eval_metric='logloss', n_jobs=-1
        )
    }

    # Weights based on typical performance
    weights = {
        'logistic': 0.2,
        'random_forest': 0.35,
        'xgboost': 0.45
    }

    ensemble = ModelEnsemble(models=models, voting='soft', weights=weights)
    ensemble.fit(X_train, y_train)

    return ensemble
