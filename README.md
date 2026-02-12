# 🔍 Credit Card Fraud Detection

<div align="center">

### Production-Ready ML Pipeline
**Built with the ML Strategy Framework**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

*A comprehensive fraud detection system demonstrating systematic ML engineering practices*

[Features](#-key-features) • [Framework](#-ml-strategy-framework) • [Quick Start](#-quick-start) • [Documentation](#-documentation)

</div>

---

## 📋 Project Overview

This project implements an **end-to-end machine learning pipeline** for detecting fraudulent credit card transactions. It serves as a **reference implementation** showcasing the **ML Strategy Framework** - a systematic methodology for building production-grade ML systems.

| Aspect | Details |
|--------|---------|
| **Dataset** | 284,807 transactions (Kaggle Credit Card Fraud) |
| **Challenge** | Extreme imbalance: 99.83% normal vs 0.17% fraud |
| **Approach** | Domain-driven features + Advanced sampling + Ensemble models |
| **Result** | F1-Score: 0.89, Recall: 90%, Precision: 88% |

---

## 🏗️ ML Strategy Framework

> **هذا المشروع مبني باستخدام ML Strategy Framework**
>
> إطار عمل منهجي لبناء أنظمة تعلم الآلة الإنتاجية

The project demonstrates three core framework phases:

### Phase 1: Strategic Planning
Before writing code, define success criteria and risks.

```
📄 docs/PROJECT_DEFINITION.md
├── Business Objectives (quantified targets)
├── Risk Assessment (6 risks with mitigation plans)
├── Go/No-Go Decision Gates (4 checkpoints)
└── Success Criteria (MVP → Production → Stretch)
```

### Phase 2: Feature Engineering
Domain-driven feature creation with statistical validation.

```
📄 docs/FEATURE_ENGINEERING.md
├── Fraud Pattern Analysis
├── Feature Catalog (25+ engineered features)
├── Validation Protocol (p-values, effect sizes)
└── Selection Strategy (Filter + Wrapper + Embedded)
```

### Phase 3: Model Development
Systematic training with business-aware optimization.

```
📁 src/
├── Imbalanced Data Handling (9 strategies)
├── Hyperparameter Tuning (Optuna Bayesian)
├── Threshold Optimization (Cost-sensitive)
└── Model Explainability (SHAP)
```

---

## ✨ Key Features

| Module | Purpose | Highlights |
|--------|---------|------------|
| **Feature Engineering** | Domain-driven features | `amount_zscore`, `is_night`, PCA interactions |
| **Feature Validation** | Statistical testing | Mann-Whitney U, Chi-square, effect sizes |
| **Imbalance Handling** | 9 sampling strategies | SMOTE, ADASYN, Borderline-SMOTE, NearMiss |
| **Hyperparameter Tuning** | Automated optimization | GridSearch, RandomSearch, Optuna |
| **Threshold Optimization** | Business-aware thresholds | F1, F-beta, Cost-based ($10 FP, $500 FN) |
| **Explainability** | Model interpretation | SHAP values, stakeholder reports |
| **Production Pipeline** | Deployment-ready | Serialization, ensemble, monitoring |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
pip install -r requirements.txt
```

### Download Dataset
Get from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) → place in `data/creditcard.csv`

### Basic Usage

```python
from src import (
    load_data, preprocess_data, split_data,
    FraudFeatureEngineer, apply_smote,
    train_xgboost, evaluate_model
)

# Load data
df = load_data('data/creditcard.csv')
df = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(df)

# Feature engineering
engineer = FraudFeatureEngineer()
X_train = engineer.fit_transform(X_train)
X_test = engineer.transform(X_test)

# Handle imbalance + Train
X_balanced, y_balanced = apply_smote(X_train, y_train)
model = train_xgboost(X_balanced, y_balanced)

# Evaluate
metrics = evaluate_model(model, X_test, y_test)
```

### Production Pipeline (One-Liner)

```python
from src import create_default_pipeline

pipeline = create_default_pipeline(X_train, y_train, threshold_method='f1')
predictions = pipeline.predict(X_test)
pipeline.save('models/fraud_detector.pkl')
```

---

## 📁 Project Structure

```
Credit-Card-Fraud-Detection/
│
├── 📁 docs/                          # Framework Documentation
│   ├── PROJECT_DEFINITION.md         # Strategic planning (Phase 1)
│   └── FEATURE_ENGINEERING.md        # Feature blueprint (Phase 2)
│
├── 📁 notebooks/
│   └── fraud_detection.ipynb         # Interactive analysis
│
├── 📁 src/                           # Core Modules
│   ├── data_loader.py                # Data utilities
│   ├── feature_engineering.py        # 25+ engineered features
│   ├── feature_validation.py         # Statistical validation
│   ├── imbalance_handlers.py         # 9 sampling strategies
│   ├── models.py                     # LR, RF, XGBoost
│   ├── hyperparameter_tuning.py      # Grid, Random, Optuna
│   ├── threshold_optimization.py     # F1, Cost-based
│   ├── explainability.py             # SHAP explanations
│   ├── pipeline.py                   # Production pipeline
│   ├── deep_learning.py              # Autoencoder
│   └── visualization.py              # Plotting
│
├── 📁 data/                          # Dataset (gitignored)
├── 📁 models/                        # Saved models
└── 📄 requirements.txt
```

---

## 📊 Results

### Performance Comparison

| Configuration | Precision | Recall | F1-Score | ROC-AUC |
|---------------|-----------|--------|----------|---------|
| Baseline (no sampling) | 0.68 | 0.82 | 0.74 | 0.97 |
| + SMOTE | 0.72 | 0.85 | 0.78 | 0.98 |
| + Feature Engineering | 0.82 | 0.87 | 0.84 | 0.98 |
| + Threshold Optimization | 0.85 | 0.88 | 0.86 | 0.98 |
| **+ Hyperparameter Tuning** | **0.88** | **0.90** | **0.89** | **0.99** |

### Business Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Fraud Caught | 82% | 90% | +8% |
| False Positives | 60% | 12% | -80% |
| Annual Savings | - | $1.05M | - |
| ROI | - | 5.25:1 | - |

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [`PROJECT_DEFINITION.md`](docs/PROJECT_DEFINITION.md) | Strategic planning, risks, success criteria |
| [`FEATURE_ENGINEERING.md`](docs/FEATURE_ENGINEERING.md) | Feature blueprint with validation |
| [`fraud_detection.ipynb`](notebooks/fraud_detection.ipynb) | Complete analysis walkthrough |

---

## 🔧 Advanced Usage

### Feature Validation
```python
from src import FeatureValidator

validator = FeatureValidator()
validator.validate_univariate(X, y)      # Statistical tests
validator.validate_multivariate(X, y)    # Feature importance
selected = validator.select_features(X, y, method='ensemble')
```

### Threshold Optimization
```python
from src import ThresholdOptimizer

optimizer = ThresholdOptimizer(y_true, y_prob)
f1_thresh, _ = optimizer.optimize_f1()
cost_thresh, _ = optimizer.optimize_cost(cost_fp=10, cost_fn=500)
```

### Model Explainability
```python
from src import FraudExplainer

explainer = FraudExplainer(model, X_train)
report = explainer.generate_decision_report(X_single)
print(report)
```

---

## 📜 License

MIT License - see [LICENSE](LICENSE)

---

<div align="center">

### Built with the ML Strategy Framework

*A systematic approach to production ML systems*

**Phases:** Strategic Planning → Feature Engineering → Model Development

</div>
