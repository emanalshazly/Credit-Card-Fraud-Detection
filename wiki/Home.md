# Credit Card Fraud Detection - Wiki

<div align="center">

**Production-Ready ML Pipeline Built with the ML Strategy Framework**

[![Author](https://img.shields.io/badge/Author-Iman%20Elshazli-blue)](https://www.linkedin.com/in/monna1478/)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Status](https://img.shields.io/badge/Status-Production--Ready-green.svg)

</div>

---

## Welcome

This wiki provides comprehensive documentation for the **Credit Card Fraud Detection** project - a showcase implementation of the **ML Strategy Framework** for building production-grade machine learning systems.

### What Makes This Project Special?

| Aspect | Description |
|--------|-------------|
| **Framework-Driven** | Built using a systematic ML Strategy Framework methodology |
| **Production-Ready** | Complete pipeline from data to deployment |
| **Comprehensive** | 25+ engineered features, 9 sampling strategies, 3 optimization methods |
| **Explainable** | SHAP-based model interpretation for stakeholders |

---

## Quick Navigation

### Getting Started
- [[Getting Started]] - Installation, setup, and first run
- [[Quick Start Guide]] - 5-minute introduction

### The Framework
- [[ML Strategy Framework]] - Understanding the methodology
- [[Project Definition]] - Strategic planning phase
- [[Feature Engineering Guide]] - Domain-driven features

### Technical Guides
- [[Model Training]] - Training and optimization
- [[Imbalanced Data Handling]] - SMOTE, ADASYN, and more
- [[Threshold Optimization]] - Business-aware thresholds

### Reference
- [[API Reference]] - Complete code documentation
- [[FAQ]] - Frequently asked questions
- [[Troubleshooting]] - Common issues and solutions

---

## Project Overview

### The Challenge

Credit card fraud detection presents a classic **imbalanced classification** problem:

```
Total Transactions: 284,807
├── Normal (Class 0): 284,315 (99.83%)
└── Fraud (Class 1):      492 (0.17%)

Imbalance Ratio: 578:1
```

### Our Approach

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML STRATEGY FRAMEWORK                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: STRATEGIC PLANNING                                    │
│  ├── Define business objectives                                 │
│  ├── Quantify success criteria                                  │
│  └── Identify risks & mitigation                                │
│                                                                 │
│  Phase 2: FEATURE ENGINEERING                                   │
│  ├── Domain analysis                                            │
│  ├── Feature creation (25+ features)                            │
│  └── Statistical validation                                     │
│                                                                 │
│  Phase 3: MODEL DEVELOPMENT                                     │
│  ├── Imbalance handling (9 strategies)                          │
│  ├── Hyperparameter tuning (Optuna)                             │
│  ├── Threshold optimization                                     │
│  └── Model explainability (SHAP)                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Results

| Metric | Baseline | Final Model | Improvement |
|--------|----------|-------------|-------------|
| Precision | 0.68 | 0.88 | +29% |
| Recall | 0.82 | 0.90 | +10% |
| F1-Score | 0.74 | 0.89 | +20% |
| ROC-AUC | 0.97 | 0.99 | +2% |

---

## Architecture

```
Credit-Card-Fraud-Detection/
│
├── docs/                          # Framework Documentation
│   ├── PROJECT_DEFINITION.md      # Phase 1: Strategic Planning
│   └── FEATURE_ENGINEERING.md     # Phase 2: Feature Blueprint
│
├── wiki/                          # This Wiki
│
├── notebooks/
│   └── fraud_detection.ipynb      # Interactive Analysis
│
├── src/                           # Core Modules
│   ├── data_loader.py             # Data utilities
│   ├── feature_engineering.py     # Feature creation
│   ├── feature_validation.py      # Statistical validation
│   ├── imbalance_handlers.py      # 9 sampling strategies
│   ├── models.py                  # LR, RF, XGBoost
│   ├── hyperparameter_tuning.py   # Optimization
│   ├── threshold_optimization.py  # Business thresholds
│   ├── explainability.py          # SHAP explanations
│   ├── pipeline.py                # Production pipeline
│   ├── deep_learning.py           # Autoencoder
│   └── visualization.py           # Plotting
│
├── data/                          # Dataset (gitignored)
├── models/                        # Saved models
└── requirements.txt
```

---

## Key Features

### 1. Domain-Driven Feature Engineering
- 25+ engineered features based on fraud patterns
- Statistical validation with Mann-Whitney U, Chi-square tests
- Effect size calculation for feature importance

### 2. Advanced Imbalance Handling
Nine sampling strategies to address 578:1 imbalance:
- SMOTE, ADASYN, Borderline-SMOTE
- SMOTE-ENN, SMOTE-Tomek
- Random Undersampling, NearMiss
- Cluster Centroids
- Class Weights

### 3. Business-Aware Optimization
- Cost-sensitive threshold tuning ($10 FP vs $500 FN)
- Multiple optimization methods (F1, F-beta, Youden's J)
- Stakeholder-ready decision reports

### 4. Model Explainability
- SHAP value analysis
- Feature importance visualization
- Transaction-level explanations

---

## Author

**Iman Elshazli** - Prompt Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue.svg)](https://www.linkedin.com/in/monna1478/)

---

## License

This project is licensed under the MIT License.

---

<div align="center">

**Built with the ML Strategy Framework**

*A systematic approach to production ML systems*

</div>
