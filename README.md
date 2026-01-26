# Credit Card Fraud Detection

## الهدف (Objective)
تحديد المعاملات الاحتيالية - Identify fraudulent credit card transactions using machine learning.

## Dataset
- **Source**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Features**: 30 (V1-V28 from PCA, Time, Amount)
- **Target**: Class (0 = Normal, 1 = Fraud)

## Challenge: Imbalanced Dataset
- **Normal Transactions**: 99.83%
- **Fraud Transactions**: 0.17%
- **Imbalance Ratio**: ~578:1

## Solutions Implemented

### 1. SMOTE (Synthetic Minority Over-sampling Technique)
Creates synthetic samples of the minority class to balance the dataset.

### 2. Random Undersampling
Reduces the majority class samples to match the minority class.

### 3. Class Weights
Assigns higher weights to the minority class during model training.

## Models

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Linear classifier with probability outputs |
| **Random Forest** | Ensemble of decision trees |
| **XGBoost** | Gradient boosting algorithm |

## Project Structure

```
Credit-Card-Fraud-Detection/
├── data/                    # Dataset (download from Kaggle)
│   └── creditcard.csv
├── models/                  # Saved models
├── notebooks/
│   └── fraud_detection.ipynb  # Main analysis notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Data loading utilities
│   ├── imbalance_handlers.py # SMOTE, Undersampling, Weights
│   ├── models.py            # Model training & evaluation
│   └── visualization.py     # Plotting functions
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone repository
git clone https://github.com/emanalshazly/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle and place in data/
```

## Usage

```python
from src import load_data, preprocess_data, split_data
from src import apply_smote, get_class_weights
from src import train_xgboost, evaluate_model

# Load and preprocess
df = load_data('data/creditcard.csv')
df = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(df)

# Handle imbalance
X_resampled, y_resampled = apply_smote(X_train, y_train)

# Train and evaluate
model = train_xgboost(X_resampled, y_resampled)
metrics = evaluate_model(model, X_test, y_test)
```

## Evaluation Metrics

For imbalanced classification, we focus on:
- **Recall**: Percentage of actual frauds detected
- **Precision**: Percentage of predicted frauds that are actual frauds
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Average Precision**: Area under the Precision-Recall curve

## Key Findings

1. **Baseline models** have high accuracy but poor fraud detection (low recall)
2. **SMOTE** significantly improves recall with some precision trade-off
3. **Class weights** provide the best balance for XGBoost
4. **XGBoost** outperforms other models in most scenarios

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- xgboost >= 1.7.0
- imbalanced-learn >= 0.10.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
