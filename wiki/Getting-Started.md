# Getting Started

This guide will help you set up and run the Credit Card Fraud Detection project.

---

## Prerequisites

- Python 3.9 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)
- Kaggle account (for dataset download)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/emanalshazly/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

1. Go to [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place it in the `data/` folder

```
Credit-Card-Fraud-Detection/
└── data/
    └── creditcard.csv  <- Place file here
```

---

## Quick Verification

Run this Python code to verify your setup:

```python
import pandas as pd
from src import load_data, preprocess_data

# Load and verify
df = load_data('data/creditcard.csv')
print(f"Dataset loaded: {len(df):,} transactions")
print(f"Fraud cases: {df['Class'].sum():,} ({df['Class'].mean()*100:.2f}%)")
```

Expected output:
```
Dataset loaded: 284,807 transactions
Fraud cases: 492 (0.17%)
```

---

## Your First Model

### Basic Usage (5 minutes)

```python
from src import (
    load_data, preprocess_data, split_data,
    apply_smote, train_xgboost, evaluate_model
)

# 1. Load data
df = load_data('data/creditcard.csv')
df = preprocess_data(df)

# 2. Split
X_train, X_test, y_train, y_test = split_data(df)

# 3. Handle imbalance
X_balanced, y_balanced = apply_smote(X_train, y_train)

# 4. Train
model = train_xgboost(X_balanced, y_balanced)

# 5. Evaluate
metrics = evaluate_model(model, X_test, y_test)
print(f"F1-Score: {metrics['f1']:.3f}")
```

### With Feature Engineering (10 minutes)

```python
from src import (
    load_data, preprocess_data, split_data,
    FraudFeatureEngineer, apply_smote,
    train_xgboost, evaluate_model
)

# 1. Load data
df = load_data('data/creditcard.csv')
df = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(df)

# 2. Feature engineering
engineer = FraudFeatureEngineer()
X_train_fe = engineer.fit_transform(X_train)
X_test_fe = engineer.transform(X_test)

# 3. Handle imbalance
X_balanced, y_balanced = apply_smote(X_train_fe, y_train)

# 4. Train
model = train_xgboost(X_balanced, y_balanced)

# 5. Evaluate
metrics = evaluate_model(model, X_test_fe, y_test)
print(f"F1-Score: {metrics['f1']:.3f}")
```

### Production Pipeline (One-Liner)

```python
from src import create_default_pipeline

# Complete pipeline
pipeline = create_default_pipeline(X_train, y_train, threshold_method='f1')

# Predict
predictions = pipeline.predict(X_test)

# Save for deployment
pipeline.save('models/fraud_detector.pkl')
```

---

## Using the Jupyter Notebook

For interactive exploration:

```bash
jupyter notebook notebooks/fraud_detection.ipynb
```

The notebook includes:
- Exploratory Data Analysis
- Feature Engineering walkthrough
- Model comparison
- Visualization

---

## Project Structure

```
Credit-Card-Fraud-Detection/
│
├── data/                     # Dataset (you add this)
│   └── creditcard.csv
│
├── docs/                     # Documentation
│   ├── PROJECT_DEFINITION.md # Strategic planning
│   └── FEATURE_ENGINEERING.md# Feature blueprint
│
├── wiki/                     # This wiki
│
├── notebooks/
│   └── fraud_detection.ipynb # Interactive notebook
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── feature_validation.py
│   ├── imbalance_handlers.py
│   ├── models.py
│   ├── hyperparameter_tuning.py
│   ├── threshold_optimization.py
│   ├── explainability.py
│   ├── pipeline.py
│   ├── deep_learning.py
│   └── visualization.py
│
├── models/                   # Saved models
├── requirements.txt
└── README.md
```

---

## Common Setup Issues

### Issue: ModuleNotFoundError

```bash
# Ensure you're in project root
cd Credit-Card-Fraud-Detection

# Install in development mode
pip install -e .
```

### Issue: Memory Error

```python
# Use smaller sample for testing
df_sample = df.sample(n=50000, random_state=42)
```

### Issue: Dataset Not Found

```python
# Check path
import os
print(os.path.exists('data/creditcard.csv'))  # Should be True
```

---

## Next Steps

1. **[[ML Strategy Framework]]** - Understand the methodology
2. **[[Feature Engineering Guide]]** - Learn about features
3. **[[Model Training]]** - Advanced training options
4. **[[API Reference]]** - Full code documentation

---

## Need Help?

- Check the [[FAQ]] for common questions
- See [[Troubleshooting]] for issues
- Open an issue on GitHub

---

<div align="center">

**Author:** [Iman Elshazli](https://www.linkedin.com/in/monna1478/)

</div>
