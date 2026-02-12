# Credit Card Fraud Detection Package
# Comprehensive ML Pipeline for Fraud Detection

# Data Loading & Preprocessing
from .data_loader import load_data, preprocess_data, split_data, get_data_summary

# Imbalanced Data Handling
from .imbalance_handlers import (
    apply_smote, apply_undersampling, get_class_weights,
    apply_smote_tomek, apply_adasyn, apply_borderline_smote,
    apply_smote_enn, apply_nearmiss, compare_all_sampling_methods
)

# Feature Engineering
from .feature_engineering import FraudFeatureEngineer, create_risk_score

# Feature Validation
from .feature_validation import FeatureValidator, quick_feature_validation

# Models
from .models import (
    train_logistic_regression, train_random_forest, train_xgboost,
    evaluate_model, compare_models, save_model, load_model
)

# Hyperparameter Tuning
from .hyperparameter_tuning import (
    HyperparameterTuner, cross_validate_model, quick_tune
)

# Threshold Optimization
from .threshold_optimization import (
    ThresholdOptimizer, get_prediction_categories, optimize_threshold_quick
)

# Model Explainability
from .explainability import FraudExplainer, explain_model_quick

# Production Pipeline
from .pipeline import (
    FraudDetectionPipeline, ModelEnsemble,
    create_default_pipeline, create_ensemble_pipeline
)

# Deep Learning (optional - requires TensorFlow)
try:
    from .deep_learning import (
        FraudAutoencoder, FraudNeuralNetwork,
        train_autoencoder, train_neural_network
    )
except ImportError:
    pass  # TensorFlow not installed

# Visualization
from .visualization import (
    plot_class_distribution, plot_confusion_matrix, plot_roc_curves,
    plot_precision_recall_curves, plot_feature_importance, plot_metrics_comparison
)

__version__ = "2.0.0"
__author__ = "Fraud Detection Team"
