"""
Model Explainability for Credit Card Fraud Detection.

Interpretability methods:
1. SHAP (SHapley Additive exPlanations) - Global and Local
2. Feature Importance Analysis
3. Partial Dependence Plots
4. Decision explanation for stakeholders
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, List, Dict, Optional

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")


class FraudExplainer:
    """
    Comprehensive model explainability for fraud detection.

    Provides:
    1. Global explanations (feature importance, SHAP summary)
    2. Local explanations (individual prediction breakdown)
    3. Stakeholder-friendly decision explanations
    """

    def __init__(self, model: Any, X_train: pd.DataFrame,
                 feature_names: List[str] = None):
        """
        Initialize explainer.

        Args:
            model: Trained model (XGBoost, RandomForest, etc.)
            X_train: Training data (for background)
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or list(X_train.columns)

        # Initialize SHAP explainer
        self.shap_explainer = None
        self.shap_values = None

        if SHAP_AVAILABLE:
            self._init_shap_explainer()

    def _init_shap_explainer(self):
        """Initialize appropriate SHAP explainer based on model type."""
        model_type = type(self.model).__name__

        try:
            if 'XGB' in model_type or 'LightGBM' in model_type or 'CatBoost' in model_type:
                # Tree-based models
                self.shap_explainer = shap.TreeExplainer(self.model)
            elif 'RandomForest' in model_type or 'GradientBoosting' in model_type:
                # Sklearn tree models
                self.shap_explainer = shap.TreeExplainer(self.model)
            elif 'Logistic' in model_type or 'Linear' in model_type:
                # Linear models
                self.shap_explainer = shap.LinearExplainer(
                    self.model, self.X_train
                )
            else:
                # Fallback to KernelExplainer (slower but universal)
                background = shap.sample(self.X_train, 100)
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba, background
                )
        except Exception as e:
            print(f"Warning: Could not initialize SHAP explainer: {e}")
            self.shap_explainer = None

    def compute_shap_values(self, X: pd.DataFrame = None,
                           max_samples: int = 1000) -> np.ndarray:
        """
        Compute SHAP values for dataset.

        Args:
            X: Data to explain (default: training data sample)
            max_samples: Maximum samples to compute (for performance)

        Returns:
            Array of SHAP values
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            print("SHAP not available")
            return None

        if X is None:
            X = self.X_train

        # Sample if too large
        if len(X) > max_samples:
            X = X.sample(n=max_samples, random_state=42)

        self.shap_values = self.shap_explainer.shap_values(X)

        # Handle different return formats
        if isinstance(self.shap_values, list):
            # For classifiers, take positive class
            self.shap_values = self.shap_values[1]

        return self.shap_values

    def get_global_importance(self, X: pd.DataFrame = None,
                             top_n: int = 20) -> pd.DataFrame:
        """
        Get global feature importance using mean absolute SHAP values.

        Args:
            X: Data to explain
            top_n: Number of top features

        Returns:
            DataFrame with feature importance
        """
        if X is None:
            X = self.X_train

        if self.shap_values is None:
            self.compute_shap_values(X)

        if self.shap_values is None:
            # Fallback to model's feature importance
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
            else:
                return None

            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        else:
            # Use SHAP values
            mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
            importance_df = pd.DataFrame({
                'feature': self.feature_names[:len(mean_abs_shap)],
                'importance': mean_abs_shap
            }).sort_values('importance', ascending=False)

        return importance_df.head(top_n)

    def explain_prediction(self, X_single: pd.DataFrame,
                          threshold: float = 0.5) -> Dict:
        """
        Explain a single prediction in human-readable format.

        Args:
            X_single: Single transaction (1 row DataFrame)
            threshold: Classification threshold

        Returns:
            Dictionary with explanation
        """
        # Get prediction
        prob = self.model.predict_proba(X_single)[0, 1]
        prediction = 'FRAUD' if prob >= threshold else 'NORMAL'

        # Get SHAP values for this instance
        if SHAP_AVAILABLE and self.shap_explainer is not None:
            shap_vals = self.shap_explainer.shap_values(X_single)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            shap_vals = shap_vals[0]

            # Get top contributing features
            feature_contributions = pd.DataFrame({
                'feature': self.feature_names[:len(shap_vals)],
                'contribution': shap_vals,
                'value': X_single.values[0][:len(shap_vals)]
            }).sort_values('contribution', key=abs, ascending=False)

            top_positive = feature_contributions[feature_contributions['contribution'] > 0].head(5)
            top_negative = feature_contributions[feature_contributions['contribution'] < 0].head(5)
        else:
            top_positive = None
            top_negative = None
            feature_contributions = None

        explanation = {
            'prediction': prediction,
            'probability': prob,
            'confidence': 'HIGH' if prob > 0.8 or prob < 0.2 else 'MEDIUM' if prob > 0.6 or prob < 0.4 else 'LOW',
            'top_fraud_indicators': top_positive.to_dict('records') if top_positive is not None else [],
            'top_normal_indicators': top_negative.to_dict('records') if top_negative is not None else [],
            'feature_contributions': feature_contributions
        }

        return explanation

    def generate_decision_report(self, X_single: pd.DataFrame,
                                threshold: float = 0.5) -> str:
        """
        Generate human-readable decision report for stakeholders.

        Args:
            X_single: Single transaction
            threshold: Classification threshold

        Returns:
            Formatted report string
        """
        explanation = self.explain_prediction(X_single, threshold)

        report = f"""
{'='*60}
FRAUD DETECTION DECISION REPORT
{'='*60}

DECISION: {explanation['prediction']}
Fraud Probability: {explanation['probability']:.2%}
Confidence: {explanation['confidence']}

{'='*60}
RISK FACTORS (Contributing to fraud score):
{'='*60}
"""
        if explanation['top_fraud_indicators']:
            for i, factor in enumerate(explanation['top_fraud_indicators'], 1):
                report += f"{i}. {factor['feature']}: {factor['value']:.4f} (contribution: +{factor['contribution']:.4f})\n"
        else:
            report += "Unable to compute detailed risk factors.\n"

        report += f"""
{'='*60}
POSITIVE FACTORS (Reducing fraud score):
{'='*60}
"""
        if explanation['top_normal_indicators']:
            for i, factor in enumerate(explanation['top_normal_indicators'], 1):
                report += f"{i}. {factor['feature']}: {factor['value']:.4f} (contribution: {factor['contribution']:.4f})\n"
        else:
            report += "Unable to compute detailed positive factors.\n"

        report += f"""
{'='*60}
RECOMMENDED ACTION:
{'='*60}
"""
        if explanation['probability'] >= 0.8:
            report += "AUTO-BLOCK: High confidence fraud. Block transaction immediately.\n"
        elif explanation['probability'] >= 0.5:
            report += "MANUAL REVIEW: Moderate fraud risk. Requires human review.\n"
        elif explanation['probability'] >= 0.2:
            report += "MONITOR: Low fraud risk but suspicious. Flag for monitoring.\n"
        else:
            report += "APPROVE: Low fraud risk. Transaction appears legitimate.\n"

        return report

    def plot_shap_summary(self, X: pd.DataFrame = None,
                         max_display: int = 20, figsize: tuple = (10, 8)):
        """
        Plot SHAP summary (beeswarm plot).
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available for plotting")
            return None

        if X is None:
            X = self.X_train.sample(n=min(1000, len(self.X_train)), random_state=42)

        if self.shap_values is None:
            self.compute_shap_values(X)

        plt.figure(figsize=figsize)
        shap.summary_plot(self.shap_values, X, feature_names=self.feature_names,
                         max_display=max_display, show=False)
        plt.title('SHAP Feature Importance Summary')
        plt.tight_layout()
        return plt.gcf()

    def plot_shap_waterfall(self, X_single: pd.DataFrame, figsize: tuple = (10, 6)):
        """
        Plot SHAP waterfall for single prediction.
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available for plotting")
            return None

        shap_vals = self.shap_explainer.shap_values(X_single)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        # Create Explanation object for new SHAP API
        base_value = self.shap_explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]

        explanation = shap.Explanation(
            values=shap_vals[0],
            base_values=base_value,
            data=X_single.values[0],
            feature_names=self.feature_names[:len(shap_vals[0])]
        )

        plt.figure(figsize=figsize)
        shap.waterfall_plot(explanation, show=False)
        plt.title('Prediction Explanation (Waterfall)')
        plt.tight_layout()
        return plt.gcf()

    def plot_shap_force(self, X_single: pd.DataFrame, figsize: tuple = (20, 3)):
        """
        Plot SHAP force plot for single prediction.
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available for plotting")
            return None

        shap_vals = self.shap_explainer.shap_values(X_single)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        base_value = self.shap_explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]

        # Force plot
        plt.figure(figsize=figsize)
        shap.force_plot(
            base_value,
            shap_vals[0],
            X_single.iloc[0],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        plt.title('Force Plot: Feature Contributions')
        plt.tight_layout()
        return plt.gcf()

    def plot_feature_dependence(self, feature: str, X: pd.DataFrame = None,
                               interaction_feature: str = None,
                               figsize: tuple = (10, 6)):
        """
        Plot SHAP dependence plot for specific feature.
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available for plotting")
            return None

        if X is None:
            X = self.X_train.sample(n=min(1000, len(self.X_train)), random_state=42)

        if self.shap_values is None:
            self.compute_shap_values(X)

        plt.figure(figsize=figsize)
        shap.dependence_plot(
            feature,
            self.shap_values,
            X,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False
        )
        plt.title(f'SHAP Dependence: {feature}')
        plt.tight_layout()
        return plt.gcf()


def explain_model_quick(model, X_train: pd.DataFrame,
                       X_test_sample: pd.DataFrame = None,
                       top_n: int = 15) -> Dict:
    """
    Quick model explanation.

    Args:
        model: Trained model
        X_train: Training data
        X_test_sample: Sample of test data to explain
        top_n: Number of top features

    Returns:
        Dictionary with explanation results
    """
    explainer = FraudExplainer(model, X_train)

    results = {
        'global_importance': explainer.get_global_importance(top_n=top_n)
    }

    if X_test_sample is not None and len(X_test_sample) > 0:
        # Explain first prediction
        results['sample_explanation'] = explainer.explain_prediction(
            X_test_sample.iloc[[0]]
        )
        results['sample_report'] = explainer.generate_decision_report(
            X_test_sample.iloc[[0]]
        )

    return results


def get_feature_importance_comparison(models: Dict[str, Any],
                                     X_train: pd.DataFrame,
                                     top_n: int = 15) -> pd.DataFrame:
    """
    Compare feature importance across multiple models.

    Args:
        models: Dictionary of {model_name: model}
        X_train: Training data
        top_n: Number of features

    Returns:
        DataFrame with importance comparison
    """
    all_importance = []

    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            for i, feature in enumerate(X_train.columns):
                all_importance.append({
                    'model': name,
                    'feature': feature,
                    'importance': importance[i] if i < len(importance) else 0
                })

    df = pd.DataFrame(all_importance)

    # Pivot for comparison
    pivot_df = df.pivot(index='feature', columns='model', values='importance')
    pivot_df['avg_importance'] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values('avg_importance', ascending=False).head(top_n)

    return pivot_df
