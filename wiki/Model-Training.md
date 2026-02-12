# Model Training Guide

This guide covers the complete model training pipeline, from handling imbalanced data to hyperparameter optimization and threshold tuning.

---

## Training Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. DATA PREPARATION                                           │
│   ├── Load data                                                 │
│   ├── Feature engineering                                       │
│   └── Train/test split                                          │
│                                                                 │
│   2. IMBALANCE HANDLING                                         │
│   ├── Choose sampling strategy                                  │
│   └── Apply to training data only                               │
│                                                                 │
│   3. MODEL TRAINING                                             │
│   ├── Select algorithm(s)                                       │
│   ├── Hyperparameter tuning                                     │
│   └── Cross-validation                                          │
│                                                                 │
│   4. THRESHOLD OPTIMIZATION                                     │
│   ├── Select optimization method                                │
│   └── Tune for business metrics                                 │
│                                                                 │
│   5. EVALUATION & DEPLOYMENT                                    │
│   ├── Test set evaluation                                       │
│   ├── Model explainability                                      │
│   └── Save for production                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Data Preparation

```python
from src import load_data, preprocess_data, split_data, FraudFeatureEngineer

# Load data
df = load_data('data/creditcard.csv')
df = preprocess_data(df)

# Split first (before feature engineering)
X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)

# Feature engineering
engineer = FraudFeatureEngineer()
X_train = engineer.fit_transform(X_train)
X_test = engineer.transform(X_test)

print(f"Training: {len(X_train):,} samples")
print(f"Testing: {len(X_test):,} samples")
print(f"Features: {X_train.shape[1]}")
```

---

## Step 2: Handling Imbalanced Data

### The Challenge

```
Class Distribution:
├── Class 0 (Normal): 99.83%
└── Class 1 (Fraud):   0.17%

Imbalance Ratio: 578:1
```

### Available Strategies

| Strategy | Type | Description | When to Use |
|----------|------|-------------|-------------|
| SMOTE | Oversampling | Synthetic minority samples | Default choice |
| ADASYN | Oversampling | Adaptive synthetic samples | Hard examples |
| Borderline-SMOTE | Oversampling | Focus on boundary | Decision boundaries |
| SMOTE-ENN | Combined | Oversample + clean | Noisy data |
| SMOTE-Tomek | Combined | Oversample + clean | Overlapping classes |
| Random Undersample | Undersampling | Random removal | Large datasets |
| NearMiss | Undersampling | Informed removal | Preserve structure |
| Cluster Centroids | Undersampling | Cluster-based | Large datasets |
| Class Weights | None | Algorithm-level | Quick solution |

### Using Sampling Methods

```python
from src import (
    apply_smote,
    apply_adasyn,
    apply_borderline_smote,
    apply_smote_enn,
    apply_random_undersample,
    compare_all_sampling_methods,
    get_best_sampling_method
)

# Basic SMOTE
X_balanced, y_balanced = apply_smote(X_train, y_train)

# Compare all methods
results = compare_all_sampling_methods(X_train, y_train, model)

# Get best method automatically
best_method, best_data = get_best_sampling_method(X_train, y_train, model)
```

### Comparison Example

```python
# Compare sampling methods
from src import compare_all_sampling_methods
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
results = compare_all_sampling_methods(X_train, y_train, model)

# Print results
for method, metrics in results.items():
    print(f"{method}: F1={metrics['f1']:.3f}, Recall={metrics['recall']:.3f}")
```

Expected output:
```
SMOTE: F1=0.847, Recall=0.872
ADASYN: F1=0.839, Recall=0.885
Borderline-SMOTE: F1=0.852, Recall=0.864
SMOTE-ENN: F1=0.861, Recall=0.858
Random Undersample: F1=0.812, Recall=0.891
Class Weights: F1=0.834, Recall=0.879
```

---

## Step 3: Model Training

### Available Models

| Model | Pros | Cons | Use Case |
|-------|------|------|----------|
| Logistic Regression | Fast, interpretable | Linear only | Baseline |
| Random Forest | Robust, handles non-linear | Slower | General |
| XGBoost | Best performance | Complex | Production |

### Basic Training

```python
from src import train_logistic_regression, train_random_forest, train_xgboost

# Logistic Regression
lr_model = train_logistic_regression(X_balanced, y_balanced)

# Random Forest
rf_model = train_random_forest(X_balanced, y_balanced, n_estimators=100)

# XGBoost
xgb_model = train_xgboost(X_balanced, y_balanced)
```

### With Class Weights (Alternative to Sampling)

```python
from sklearn.ensemble import RandomForestClassifier

# Calculate weights
class_weights = {0: 1, 1: 578}  # Inverse of class frequency

model = RandomForestClassifier(
    n_estimators=100,
    class_weight=class_weights,
    random_state=42
)
model.fit(X_train, y_train)
```

---

## Step 4: Hyperparameter Optimization

### Grid Search

```python
from src import HyperparameterTuner

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

# Initialize tuner
tuner = HyperparameterTuner(model, param_grid)

# Grid search
best_params = tuner.grid_search(X_balanced, y_balanced, cv=5)
print(f"Best params: {best_params}")
```

### Random Search (Faster)

```python
# Random search with 50 iterations
best_params = tuner.random_search(
    X_balanced, y_balanced,
    n_iter=50,
    cv=5
)
```

### Bayesian Optimization (Smartest)

```python
# Optuna-based Bayesian optimization
best_params = tuner.optuna_search(
    X_balanced, y_balanced,
    n_trials=100,
    cv=5
)
```

### XGBoost-Specific Tuning

```python
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'scale_pos_weight': [1, 10, 100]  # Handle imbalance
}
```

---

## Step 5: Threshold Optimization

### Why Threshold Matters

Default threshold (0.5) rarely optimal for fraud detection:

```
Threshold  Precision  Recall  F1-Score
0.30       0.72       0.94    0.82
0.50       0.88       0.78    0.83
0.70       0.95       0.62    0.75
```

### Optimization Methods

```python
from src import ThresholdOptimizer

# Get probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Initialize optimizer
optimizer = ThresholdOptimizer(y_test, y_prob)

# Method 1: Maximize F1
threshold_f1, metrics_f1 = optimizer.optimize_f1()
print(f"F1-optimal threshold: {threshold_f1:.3f}")

# Method 2: Cost-sensitive
# $10 cost per false positive, $500 cost per false negative
threshold_cost, metrics_cost = optimizer.optimize_cost(
    cost_fp=10,
    cost_fn=500
)
print(f"Cost-optimal threshold: {threshold_cost:.3f}")

# Method 3: Youden's J statistic (balance sensitivity/specificity)
threshold_youden, metrics_youden = optimizer.optimize_youden_j()

# Method 4: F-beta (emphasize recall)
threshold_fbeta, metrics_fbeta = optimizer.optimize_fbeta(beta=2)
```

### Visualize Threshold Impact

```python
# Plot threshold vs metrics
optimizer.plot_threshold_analysis()
```

---

## Step 6: Evaluation

### Comprehensive Evaluation

```python
from src import evaluate_model

# Evaluate with optimized threshold
metrics = evaluate_model(
    model,
    X_test,
    y_test,
    threshold=threshold_f1
)

print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1-Score: {metrics['f1']:.3f}")
print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
```

### Confusion Matrix Analysis

```
                 Predicted
                 Normal  Fraud
Actual Normal    56,850    114
       Fraud         10     88

True Positives:  88 (fraud caught)
False Positives: 114 (legitimate flagged)
False Negatives: 10 (fraud missed)
True Negatives:  56,850 (legitimate passed)
```

### Business Metrics

```python
# Calculate business impact
fp_cost = metrics['false_positives'] * 10   # $10 per FP
fn_cost = metrics['false_negatives'] * 500  # $500 per FN
total_cost = fp_cost + fn_cost

print(f"False Positive Cost: ${fp_cost:,}")
print(f"False Negative Cost: ${fn_cost:,}")
print(f"Total Cost: ${total_cost:,}")
```

---

## Complete Training Example

```python
from src import (
    load_data, preprocess_data, split_data,
    FraudFeatureEngineer, apply_smote,
    HyperparameterTuner, ThresholdOptimizer,
    evaluate_model, FraudExplainer,
    create_default_pipeline
)
from xgboost import XGBClassifier

# 1. Prepare data
df = load_data('data/creditcard.csv')
df = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(df)

# 2. Feature engineering
engineer = FraudFeatureEngineer()
X_train = engineer.fit_transform(X_train)
X_test = engineer.transform(X_test)

# 3. Handle imbalance
X_balanced, y_balanced = apply_smote(X_train, y_train)

# 4. Hyperparameter tuning
model = XGBClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 7],
    'learning_rate': [0.1]
}
tuner = HyperparameterTuner(model, param_grid)
best_params = tuner.grid_search(X_balanced, y_balanced)

# 5. Train final model
final_model = XGBClassifier(**best_params, random_state=42)
final_model.fit(X_balanced, y_balanced)

# 6. Optimize threshold
y_prob = final_model.predict_proba(X_test)[:, 1]
optimizer = ThresholdOptimizer(y_test, y_prob)
best_threshold, _ = optimizer.optimize_f1()

# 7. Evaluate
metrics = evaluate_model(final_model, X_test, y_test, threshold=best_threshold)
print(f"Final F1-Score: {metrics['f1']:.3f}")

# 8. Explain
explainer = FraudExplainer(final_model, X_train)
explainer.get_feature_importance()

# 9. Save for production
import joblib
joblib.dump({
    'model': final_model,
    'engineer': engineer,
    'threshold': best_threshold
}, 'models/fraud_detector.pkl')
```

---

## Next Steps

- [[Imbalanced Data Handling]] - Deep dive into sampling
- [[API Reference]] - Full code documentation
- [[FAQ]] - Common questions

---

<div align="center">

**Author:** [Iman Elshazli](https://www.linkedin.com/in/monna1478/)

</div>
