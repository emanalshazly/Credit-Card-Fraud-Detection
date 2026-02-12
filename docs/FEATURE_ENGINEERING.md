# 🔧 Feature Engineering Blueprint

## Credit Card Fraud Detection

> **Part of the ML Strategy Framework - Phase 2**

---

## 1. Domain Analysis

### Fraud Patterns Identified

| Pattern | Description | Feature Strategy |
|---------|-------------|------------------|
| **Velocity Fraud** | Multiple rapid transactions to max out card | `transaction_freq_1h`, `time_since_last` |
| **Amount Anomalies** | Unusual amounts (too large, round numbers) | `amount_zscore`, `amount_is_round` |
| **Temporal Anomalies** | Late night transactions when victim asleep | `is_night`, `hour_of_day` |
| **Behavioral Shift** | Sudden change from normal patterns | `amount_vs_median`, `v_outlier_count` |

---

## 2. Feature Catalog

### Tier 1: Amount-Based Features

| Feature | Business Logic | Implementation |
|---------|---------------|----------------|
| `amount_zscore` | How unusual is this amount globally? | `(Amount - mean) / std` |
| `amount_log` | Handle skewness in amounts | `log1p(Amount)` |
| `amount_is_round` | Fraudsters use round numbers ($50, $100) | `Amount % 10 == 0` |
| `amount_is_high` | Top 5% amounts are suspicious | `Amount > quantile(0.95)` |
| `amount_vs_median` | Ratio to global median | `Amount / median` |
| `amount_is_small` | Card testing pattern | `Amount < 10` |

### Tier 2: Time-Based Features

| Feature | Business Logic | Implementation |
|---------|---------------|----------------|
| `hour_of_day` | Extract hour (0-23) | `(Time % 86400) / 3600` |
| `is_night` | Fraud higher at night (12am-6am) | `hour in [0, 6]` |
| `is_business_hours` | Normal transactions (9am-5pm) | `hour in [9, 17]` |
| `time_bin` | Morning/Afternoon/Evening/Night | Categorical bins |

### Tier 3: Statistical Features

| Feature | Business Logic | Implementation |
|---------|---------------|----------------|
| `v_sum_abs` | Overall anomaly score | `sum(abs(V1..V28))` |
| `v_mean` | Average PCA component value | `mean(V1..V28)` |
| `v_std` | Variability across components | `std(V1..V28)` |
| `v_outlier_count` | Number of extreme values | `count(abs(V) > 2)` |
| `v_skew` | Distribution skewness | `skew(V1..V28)` |
| `v_kurtosis` | Distribution peakedness | `kurtosis(V1..V28)` |

### Tier 4: Interaction Features

| Feature | Business Logic | Implementation |
|---------|---------------|----------------|
| `V14_squared` | Non-linear pattern in V14 | `V14 ** 2` |
| `V17_squared` | Non-linear pattern in V17 | `V17 ** 2` |
| `V14_x_V12` | Interaction between top features | `V14 * V12` |
| `Amount_x_V14` | Amount with V14 pattern | `Amount * V14` |

---

## 3. Validation Results

### Univariate Analysis

| Feature | Test | p-value | Effect Size | Decision |
|---------|------|---------|-------------|----------|
| `v_outlier_count` | Mann-Whitney U | < 0.0001 | 1.24 (Large) | ✅ KEEP |
| `amount_zscore` | Mann-Whitney U | < 0.0001 | 0.82 (Large) | ✅ KEEP |
| `v_sum_abs` | Mann-Whitney U | < 0.0001 | 0.76 (Medium) | ✅ KEEP |
| `is_night` | Chi-square | < 0.001 | - | ✅ KEEP |
| `amount_is_round` | Chi-square | < 0.01 | - | ⚠️ CONSIDER |
| `v_mean` | Mann-Whitney U | < 0.0001 | 0.45 (Small) | ✅ KEEP |

### Feature Importance (Random Forest)

```
Top 10 Features by Importance:
1. V14          : 12.3%
2. V17          : 10.8%
3. V12          :  8.4%
4. v_outlier_count:  7.2%  ← Engineered
5. amount_zscore:  6.1%  ← Engineered
6. V10          :  5.9%
7. V14_squared  :  4.8%  ← Engineered
8. V4           :  4.5%
9. v_sum_abs    :  3.9%  ← Engineered
10. Amount      :  3.2%
```

---

## 4. Implementation

### FraudFeatureEngineer Class

```python
from src import FraudFeatureEngineer

# Initialize
engineer = FraudFeatureEngineer(
    create_interactions=True,
    create_time_features=True,
    create_amount_features=True,
    create_statistical_features=True
)

# Fit on training data (learns statistics)
X_train_engineered = engineer.fit_transform(X_train)

# Transform test data (uses learned statistics)
X_test_engineered = engineer.transform(X_test)

# Get list of created features
print(engineer.get_feature_names())
```

### Risk Score (Composite Feature)

```python
from src import create_risk_score

# Creates interpretable 0-15 risk score
risk_scores = create_risk_score(X_engineered)

# Score breakdown:
# +3: amount_zscore > 2
# +2: amount_zscore > 3
# +2: is_night
# +3: v_outlier_count > 5
# +3: amount_is_very_high
# +1: amount_is_round
```

---

## 5. Feature Selection

### Ensemble Method Results

```
Method Comparison:
├── Filter (Mutual Information): 20 features
├── Filter (F-Score):            18 features
├── Wrapper (RFE):               15 features
└── Embedded (L1 Regularization): 22 features

Consensus (2+ methods): 18 features selected
```

### Final Feature Set

**Original (28):** V1-V28

**Engineered (10):**
- `amount_zscore` ✅
- `amount_log` ✅
- `amount_is_round` ✅
- `is_night` ✅
- `v_sum_abs` ✅
- `v_outlier_count` ✅
- `V14_squared` ✅
- `V14_x_V12` ✅
- `Amount_x_V14` ✅
- `risk_score` ✅

**Total: 38 features**

---

## 6. Expected Impact

| Metric | Without Engineering | With Engineering | Improvement |
|--------|--------------------|--------------------|-------------|
| F1-Score | 0.75 | 0.85 | +13% |
| Precision | 0.68 | 0.82 | +21% |
| Recall | 0.82 | 0.87 | +6% |

---

## 7. Production Considerations

### Avoiding Data Leakage

```python
# ✅ CORRECT: Fit on train, transform on test
engineer = FraudFeatureEngineer()
X_train = engineer.fit_transform(X_train)  # Learn stats
X_test = engineer.transform(X_test)        # Apply stats

# ❌ WRONG: Fitting on full dataset
X_all = engineer.fit_transform(X_all)  # Leaks test info!
```

### Handling New Data

```python
# Engineer stores global statistics for fallback
# New transactions use global stats if user stats unavailable
```

---

<div align="center">

**ML Strategy Framework - Phase 2**

*Domain-driven features with statistical validation*

</div>
