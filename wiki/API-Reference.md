# API Reference

Complete documentation for all modules in the Credit Card Fraud Detection project.

---

## Table of Contents

- [Data Loading](#data-loading)
- [Feature Engineering](#feature-engineering)
- [Feature Validation](#feature-validation)
- [Imbalance Handlers](#imbalance-handlers)
- [Models](#models)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Threshold Optimization](#threshold-optimization)
- [Explainability](#explainability)
- [Pipeline](#pipeline)
- [Deep Learning](#deep-learning)
- [Visualization](#visualization)

---

## Data Loading

**Module:** `src/data_loader.py`

### `load_data(filepath)`

Load credit card transaction data from CSV.

```python
from src import load_data

df = load_data('data/creditcard.csv')
```

**Parameters:**
- `filepath` (str): Path to the CSV file

**Returns:**
- `pd.DataFrame`: Loaded dataset

---

### `preprocess_data(df)`

Preprocess the dataset for model training.

```python
from src import preprocess_data

df = preprocess_data(df)
```

**Parameters:**
- `df` (pd.DataFrame): Raw dataset

**Returns:**
- `pd.DataFrame`: Preprocessed dataset

---

### `split_data(df, test_size=0.2, random_state=42)`

Split data into training and test sets with stratification.

```python
from src import split_data

X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
```

**Parameters:**
- `df` (pd.DataFrame): Dataset with 'Class' column
- `test_size` (float): Fraction for test set (default: 0.2)
- `random_state` (int): Random seed (default: 42)

**Returns:**
- Tuple of (X_train, X_test, y_train, y_test)

---

## Feature Engineering

**Module:** `src/feature_engineering.py`

### `FraudFeatureEngineer`

Domain-driven feature engineering for fraud detection.

```python
from src import FraudFeatureEngineer

engineer = FraudFeatureEngineer(
    add_amount_features=True,
    add_time_features=True,
    add_v_features=True,
    add_interactions=True,
    outlier_threshold=3.0
)

# Fit and transform
X_train_fe = engineer.fit_transform(X_train)
X_test_fe = engineer.transform(X_test)

# Get feature names
feature_names = engineer.get_feature_names()
```

**Methods:**

| Method | Description |
|--------|-------------|
| `fit(X)` | Learn parameters from training data |
| `transform(X)` | Apply feature engineering |
| `fit_transform(X)` | Fit and transform in one step |
| `get_feature_names()` | Get list of feature names |

**Created Features:**
- `amount_log`, `amount_zscore`, `amount_percentile`
- `hour_of_day`, `is_night`, `time_since_start`
- `v_outlier_count`, `v_sum_abs`, `v_mean`, `v_std`
- `v1_v2_interaction`, `v1_v3_interaction`, etc.

---

## Feature Validation

**Module:** `src/feature_validation.py`

### `FeatureValidator`

Statistical validation and selection of features.

```python
from src import FeatureValidator

validator = FeatureValidator()

# Univariate tests
results = validator.validate_univariate(X, y)

# Multivariate tests
importance = validator.validate_multivariate(X, y)

# Feature selection
selected = validator.select_features(X, y, method='ensemble')
```

**Methods:**

| Method | Description |
|--------|-------------|
| `validate_univariate(X, y)` | Mann-Whitney U, Chi-square tests |
| `validate_multivariate(X, y)` | Permutation importance |
| `select_features(X, y, method)` | Filter/Wrapper/Embedded selection |

**Selection Methods:**
- `'filter'`: Statistical significance (p < 0.05)
- `'wrapper'`: Recursive Feature Elimination
- `'embedded'`: L1 regularization
- `'ensemble'`: Combine all methods

---

## Imbalance Handlers

**Module:** `src/imbalance_handlers.py`

### Oversampling Methods

```python
from src import apply_smote, apply_adasyn, apply_borderline_smote

# SMOTE
X_res, y_res = apply_smote(X_train, y_train, random_state=42)

# ADASYN
X_res, y_res = apply_adasyn(X_train, y_train, random_state=42)

# Borderline-SMOTE
X_res, y_res = apply_borderline_smote(X_train, y_train, random_state=42)
```

### Combined Methods

```python
from src import apply_smote_enn, apply_smote_tomek

# SMOTE + ENN
X_res, y_res = apply_smote_enn(X_train, y_train, random_state=42)

# SMOTE + Tomek
X_res, y_res = apply_smote_tomek(X_train, y_train, random_state=42)
```

### Undersampling Methods

```python
from src import apply_random_undersample, apply_nearmiss, apply_cluster_centroids

# Random undersampling
X_res, y_res = apply_random_undersample(X_train, y_train)

# NearMiss
X_res, y_res = apply_nearmiss(X_train, y_train, version=1)

# Cluster centroids
X_res, y_res = apply_cluster_centroids(X_train, y_train)
```

### Comparison Utilities

```python
from src import compare_all_sampling_methods, get_best_sampling_method

# Compare all methods
results = compare_all_sampling_methods(X_train, y_train, model)

# Get best automatically
best_method, (X_best, y_best) = get_best_sampling_method(X_train, y_train, model)
```

---

## Models

**Module:** `src/models.py`

### Training Functions

```python
from src import train_logistic_regression, train_random_forest, train_xgboost

# Logistic Regression
model = train_logistic_regression(X_train, y_train, C=1.0)

# Random Forest
model = train_random_forest(X_train, y_train, n_estimators=100, max_depth=10)

# XGBoost
model = train_xgboost(X_train, y_train, n_estimators=100, learning_rate=0.1)
```

### Evaluation

```python
from src import evaluate_model

metrics = evaluate_model(model, X_test, y_test, threshold=0.5)
# Returns: {'precision', 'recall', 'f1', 'roc_auc', 'confusion_matrix', ...}
```

---

## Hyperparameter Tuning

**Module:** `src/hyperparameter_tuning.py`

### `HyperparameterTuner`

```python
from src import HyperparameterTuner

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10]
}

tuner = HyperparameterTuner(model, param_grid)

# Grid search
best_params = tuner.grid_search(X, y, cv=5, scoring='f1')

# Random search
best_params = tuner.random_search(X, y, n_iter=50, cv=5)

# Bayesian optimization
best_params = tuner.optuna_search(X, y, n_trials=100, cv=5)
```

**Methods:**

| Method | Description | Speed | Quality |
|--------|-------------|-------|---------|
| `grid_search()` | Exhaustive | Slow | Best |
| `random_search()` | Random sampling | Fast | Good |
| `optuna_search()` | Bayesian | Medium | Excellent |

---

## Threshold Optimization

**Module:** `src/threshold_optimization.py`

### `ThresholdOptimizer`

```python
from src import ThresholdOptimizer

optimizer = ThresholdOptimizer(y_true, y_prob)

# Optimize for F1
threshold, metrics = optimizer.optimize_f1()

# Optimize for F-beta
threshold, metrics = optimizer.optimize_fbeta(beta=2)

# Cost-sensitive
threshold, metrics = optimizer.optimize_cost(cost_fp=10, cost_fn=500)

# Youden's J
threshold, metrics = optimizer.optimize_youden_j()

# Visualize
optimizer.plot_threshold_analysis()
```

**Methods:**

| Method | Optimizes For | Use Case |
|--------|---------------|----------|
| `optimize_f1()` | F1-score | Balanced performance |
| `optimize_fbeta(beta)` | F-beta | Emphasize recall (beta > 1) |
| `optimize_cost(fp, fn)` | Total cost | Business optimization |
| `optimize_youden_j()` | TPR - FPR | Balanced sensitivity |

---

## Explainability

**Module:** `src/explainability.py`

### `FraudExplainer`

```python
from src import FraudExplainer

explainer = FraudExplainer(model, X_train)

# Global importance
importance_df = explainer.get_feature_importance()

# Single prediction explanation
explanation = explainer.explain_prediction(X_single)

# Stakeholder report
report = explainer.generate_decision_report(X_single)
print(report)
```

**Output Example:**
```
=== FRAUD DETECTION REPORT ===
Transaction ID: TXN-12345
Risk Score: 0.89 (HIGH)

Top Contributing Factors:
1. V14 = -5.23 (contributes +0.24 to risk)
2. Amount = $2,450 (contributes +0.18 to risk)
3. V4 = 3.12 (contributes +0.12 to risk)

Recommendation: FLAG FOR REVIEW
```

---

## Pipeline

**Module:** `src/pipeline.py`

### `FraudDetectionPipeline`

```python
from src import FraudDetectionPipeline

pipeline = FraudDetectionPipeline(
    sampling_method='smote',
    model_type='xgboost',
    threshold_method='f1'
)

# Train
pipeline.fit(X_train, y_train)

# Predict
predictions = pipeline.predict(X_test)
probabilities = pipeline.predict_proba(X_test)

# Evaluate
metrics = pipeline.evaluate(X_test, y_test)

# Save/Load
pipeline.save('models/pipeline.pkl')
pipeline = FraudDetectionPipeline.load('models/pipeline.pkl')
```

### `create_default_pipeline()`

Quick pipeline creation with sensible defaults.

```python
from src import create_default_pipeline

pipeline = create_default_pipeline(X_train, y_train, threshold_method='f1')
predictions = pipeline.predict(X_test)
```

### `ModelEnsemble`

Combine multiple models.

```python
from src import ModelEnsemble

ensemble = ModelEnsemble(
    models=[lr_model, rf_model, xgb_model],
    method='voting'  # or 'stacking'
)

ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)
```

---

## Deep Learning

**Module:** `src/deep_learning.py`

### `FraudAutoencoder`

Anomaly detection using autoencoder.

```python
from src import FraudAutoencoder

autoencoder = FraudAutoencoder(
    input_dim=X_train.shape[1],
    encoding_dim=14
)

# Train on normal transactions only
X_normal = X_train[y_train == 0]
autoencoder.fit(X_normal, epochs=50, batch_size=256)

# Detect anomalies
anomaly_scores = autoencoder.get_anomaly_scores(X_test)
predictions = autoencoder.predict(X_test, threshold=0.95)
```

### `FraudNeuralNetwork`

Deep neural network classifier.

```python
from src import FraudNeuralNetwork

nn = FraudNeuralNetwork(
    input_dim=X_train.shape[1],
    hidden_layers=[64, 32, 16]
)

nn.fit(X_train, y_train, epochs=100, batch_size=512)
predictions = nn.predict(X_test)
```

---

## Visualization

**Module:** `src/visualization.py`

### Plotting Functions

```python
from src import (
    plot_class_distribution,
    plot_correlation_matrix,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance
)

# Class distribution
plot_class_distribution(y)

# Correlation matrix
plot_correlation_matrix(X)

# Confusion matrix
plot_confusion_matrix(y_true, y_pred)

# ROC curve
plot_roc_curve(y_true, y_prob)

# Precision-Recall curve
plot_precision_recall_curve(y_true, y_prob)

# Feature importance
plot_feature_importance(model, feature_names, top_n=20)
```

---

## Quick Import

All main functions available from `src`:

```python
from src import (
    # Data
    load_data, preprocess_data, split_data,

    # Feature Engineering
    FraudFeatureEngineer, FeatureValidator,

    # Imbalance
    apply_smote, apply_adasyn, apply_borderline_smote,
    apply_smote_enn, apply_random_undersample,
    compare_all_sampling_methods, get_best_sampling_method,

    # Models
    train_logistic_regression, train_random_forest,
    train_xgboost, evaluate_model,

    # Optimization
    HyperparameterTuner, ThresholdOptimizer,

    # Explainability
    FraudExplainer,

    # Pipeline
    FraudDetectionPipeline, ModelEnsemble,
    create_default_pipeline,

    # Deep Learning
    FraudAutoencoder, FraudNeuralNetwork
)
```

---

<div align="center">

**Author:** [Iman Elshazli](https://www.linkedin.com/in/monna1478/)

</div>
