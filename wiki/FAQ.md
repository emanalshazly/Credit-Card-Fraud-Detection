# Frequently Asked Questions (FAQ)

Common questions and answers about the Credit Card Fraud Detection project.

---

## General Questions

### Q: What is this project?

This is a **production-ready machine learning pipeline** for detecting fraudulent credit card transactions. It demonstrates the **ML Strategy Framework** - a systematic methodology for building ML systems.

### Q: What makes this project different from other fraud detection projects?

| Aspect | Typical Projects | This Project |
|--------|------------------|--------------|
| Planning | Jump to code | Strategic planning document |
| Features | Basic | 25+ domain-driven features |
| Imbalance | One method | 9 sampling strategies |
| Threshold | Default 0.5 | Business-optimized |
| Explainability | None | SHAP-based explanations |
| Documentation | Minimal | Comprehensive |

### Q: Who created this project?

**Iman Elshazli** - Prompt Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue.svg)](https://www.linkedin.com/in/monna1478/)

---

## Data Questions

### Q: Where can I get the dataset?

Download from [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).

### Q: Why is the dataset so imbalanced?

Real-world fraud is rare. In this dataset:
- 99.83% transactions are legitimate
- 0.17% transactions are fraudulent

This 578:1 ratio is realistic and presents a significant challenge for ML models.

### Q: What do the V1-V28 columns represent?

These are **PCA-transformed features** from the original transaction data. The original features are confidential due to privacy concerns. We can still extract value by:
- Analyzing their distributions
- Creating interaction features
- Identifying outliers

### Q: Can I use my own dataset?

Yes! The pipeline is flexible. Ensure your dataset has:
- A binary target column (0 = normal, 1 = fraud)
- Numeric features

```python
# Adapt for your data
df = pd.read_csv('your_data.csv')
df = df.rename(columns={'your_target': 'Class'})
```

---

## Technical Questions

### Q: Which sampling method should I use?

Start with **SMOTE** as your default. Then:

```python
# Compare methods
from src import compare_all_sampling_methods

results = compare_all_sampling_methods(X_train, y_train, model)
best_method = max(results, key=lambda k: results[k]['f1'])
```

General guidance:
- **SMOTE**: Good all-around choice
- **SMOTE-ENN**: If you suspect noisy labels
- **ADASYN**: If decision boundary is important
- **Class Weights**: Fastest, no data modification

### Q: Why is my recall low?

Common causes:
1. **Threshold too high**: Lower it to catch more fraud
2. **Insufficient sampling**: Try more aggressive oversampling
3. **Poor features**: Add domain-driven features
4. **Model choice**: Try ensemble methods

```python
# Optimize threshold for recall
optimizer = ThresholdOptimizer(y_true, y_prob)
threshold, _ = optimizer.optimize_fbeta(beta=2)  # Emphasizes recall
```

### Q: How do I handle the precision-recall tradeoff?

Use **cost-sensitive optimization**:

```python
optimizer = ThresholdOptimizer(y_true, y_prob)

# If missing fraud costs $500 and false alarm costs $10
threshold, _ = optimizer.optimize_cost(cost_fp=10, cost_fn=500)
```

This finds the threshold that minimizes total business cost.

### Q: What's the best model for fraud detection?

**XGBoost** typically performs best, but model choice depends on your constraints:

| Model | Pros | Best For |
|-------|------|----------|
| Logistic Regression | Fast, interpretable | Baseline, explainability |
| Random Forest | Robust | General use |
| XGBoost | Best accuracy | Production |
| Neural Network | Complex patterns | Large datasets |
| Autoencoder | Unsupervised | Novelty detection |

### Q: How do I explain predictions to stakeholders?

Use the `FraudExplainer`:

```python
from src import FraudExplainer

explainer = FraudExplainer(model, X_train)
report = explainer.generate_decision_report(X_transaction)
print(report)
```

Output:
```
=== FRAUD DETECTION REPORT ===
Transaction ID: TXN-12345
Risk Score: 0.89 (HIGH)

Top Contributing Factors:
1. V14 = -5.23 (contributes +0.24 to risk)
2. Amount = $2,450 (contributes +0.18 to risk)

Recommendation: FLAG FOR REVIEW
```

---

## Performance Questions

### Q: What metrics should I use?

For imbalanced datasets, use:

| Metric | Use | Don't Use |
|--------|-----|-----------|
| F1-Score | Primary metric | Accuracy |
| Precision | If FP matters | - |
| Recall | If FN matters | - |
| ROC-AUC | Model comparison | - |
| PR-AUC | Imbalanced data | - |

**Never use accuracy** for imbalanced data - a model predicting "no fraud" for everything gets 99.83% accuracy!

### Q: What results should I expect?

With the full pipeline:

| Metric | Expected Range |
|--------|----------------|
| Precision | 0.85 - 0.92 |
| Recall | 0.85 - 0.92 |
| F1-Score | 0.85 - 0.90 |
| ROC-AUC | 0.98 - 0.99 |

### Q: How do I improve my model?

Step-by-step:

1. **Feature engineering** (+5-10% F1)
   ```python
   engineer = FraudFeatureEngineer()
   X = engineer.fit_transform(X)
   ```

2. **Better sampling** (+2-5% F1)
   ```python
   best_method, data = get_best_sampling_method(X, y, model)
   ```

3. **Hyperparameter tuning** (+2-5% F1)
   ```python
   tuner = HyperparameterTuner(model, param_grid)
   best_params = tuner.optuna_search(X, y)
   ```

4. **Threshold optimization** (+2-5% F1)
   ```python
   optimizer = ThresholdOptimizer(y_true, y_prob)
   threshold, _ = optimizer.optimize_f1()
   ```

---

## Framework Questions

### Q: What is the ML Strategy Framework?

A systematic methodology with three phases:

1. **Strategic Planning**: Define objectives, risks, success criteria
2. **Feature Engineering**: Domain-driven features with validation
3. **Model Development**: Training, optimization, deployment

### Q: Where can I learn more about the framework?

- [[ML Strategy Framework]] - Overview
- `docs/PROJECT_DEFINITION.md` - Phase 1 example
- `docs/FEATURE_ENGINEERING.md` - Phase 2 example

### Q: Can I use the framework for other projects?

Absolutely! The framework is domain-agnostic. Use the documents as templates:

1. Copy `PROJECT_DEFINITION.md`
2. Adapt objectives and risks to your domain
3. Follow the same structure

---

## Troubleshooting

### Q: I'm getting memory errors

```python
# Use smaller sample
df_sample = df.sample(n=50000, random_state=42)

# Or process in chunks
for chunk in pd.read_csv('data.csv', chunksize=10000):
    process(chunk)
```

### Q: Import errors

```bash
# Ensure you're in project root
cd Credit-Card-Fraud-Detection

# Install dependencies
pip install -r requirements.txt
```

### Q: Model training is slow

```python
# Use random search instead of grid search
best_params = tuner.random_search(X, y, n_iter=50)

# Use fewer trees
model = XGBClassifier(n_estimators=100)  # Not 1000

# Sample your data
X_sample = X.sample(frac=0.5)
```

---

## Contact & Support

- Check [[Troubleshooting]] for more issues
- Open an issue on GitHub
- Connect on [LinkedIn](https://www.linkedin.com/in/monna1478/)

---

<div align="center">

**Author:** [Iman Elshazli](https://www.linkedin.com/in/monna1478/)

</div>
