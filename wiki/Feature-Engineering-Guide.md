# Feature Engineering Guide

This guide explains the feature engineering approach used in this project, following the ML Strategy Framework methodology.

---

## Overview

Feature engineering transforms raw data into meaningful inputs that improve model performance. In fraud detection, domain knowledge is crucial for creating predictive features.

```
Raw Features (31)          Engineered Features (25+)        Final Features
┌─────────────────┐       ┌─────────────────────────┐      ┌─────────────────┐
│ Time            │       │ is_night                │      │                 │
│ Amount          │ ───▶  │ amount_zscore           │ ───▶ │ Validated &     │
│ V1-V28 (PCA)    │       │ v_outlier_count         │      │ Selected        │
│ Class           │       │ pca_interactions        │      │ Features        │
└─────────────────┘       └─────────────────────────┘      └─────────────────┘
```

---

## Fraud Pattern Analysis

Understanding fraud patterns guides feature creation:

| Pattern | Description | Target Feature |
|---------|-------------|----------------|
| **Velocity Fraud** | Multiple rapid transactions | `transaction_freq_1h` |
| **Amount Anomaly** | Unusual transaction amounts | `amount_zscore`, `amount_log` |
| **Temporal Anomaly** | Late night transactions | `is_night`, `hour_of_day` |
| **Behavioral Shift** | Deviation from patterns | `v_outlier_count` |

---

## Feature Catalog

### Category 1: Amount-Based Features

#### `amount_log`
```python
# Log transformation to reduce skewness
df['amount_log'] = np.log1p(df['Amount'])
```
- **Rationale:** Transaction amounts are heavily skewed
- **Expected Impact:** Normalizes distribution for algorithms

#### `amount_zscore`
```python
# Standard deviation from mean
df['amount_zscore'] = (df['Amount'] - mean) / std
```
- **Rationale:** Identifies unusual amounts
- **Expected Impact:** High values indicate anomalies

#### `amount_percentile`
```python
# Percentile rank of amount
df['amount_percentile'] = df['Amount'].rank(pct=True)
```
- **Rationale:** Relative position matters
- **Expected Impact:** Captures extreme transactions

---

### Category 2: Temporal Features

#### `hour_of_day`
```python
# Extract hour (0-23)
df['hour_of_day'] = (df['Time'] % 86400) / 3600
```
- **Rationale:** Fraud patterns vary by hour
- **Expected Impact:** Captures daily cycles

#### `is_night`
```python
# Flag for nighttime transactions (10 PM - 6 AM)
df['is_night'] = df['hour_of_day'].apply(
    lambda x: 1 if (x >= 22 or x < 6) else 0
)
```
- **Rationale:** Fraud more common at night
- **Expected Impact:** Strong fraud indicator

#### `time_since_start`
```python
# Normalized time from start
df['time_since_start'] = df['Time'] / df['Time'].max()
```
- **Rationale:** Time trends in data
- **Expected Impact:** Captures temporal patterns

---

### Category 3: PCA-Based Features

The dataset contains V1-V28 (PCA components). We engineer additional features from these:

#### `v_outlier_count`
```python
# Count of V components that are outliers
def count_outliers(row, thresholds):
    count = 0
    for col, (low, high) in thresholds.items():
        if row[col] < low or row[col] > high:
            count += 1
    return count

df['v_outlier_count'] = df.apply(count_outliers, axis=1)
```
- **Rationale:** Multiple anomalies suggest fraud
- **Expected Impact:** High predictive power

#### `v_sum_abs`
```python
# Sum of absolute V values
v_cols = [f'V{i}' for i in range(1, 29)]
df['v_sum_abs'] = df[v_cols].abs().sum(axis=1)
```
- **Rationale:** Aggregate deviation measure
- **Expected Impact:** Captures overall anomaly level

#### PCA Interactions
```python
# Interactions between key components
df['v1_v2_interaction'] = df['V1'] * df['V2']
df['v1_v3_interaction'] = df['V1'] * df['V3']
df['v3_v4_interaction'] = df['V3'] * df['V4']
```
- **Rationale:** Non-linear relationships
- **Expected Impact:** Captures complex patterns

---

### Category 4: Statistical Features

#### `v_mean`
```python
df['v_mean'] = df[v_cols].mean(axis=1)
```

#### `v_std`
```python
df['v_std'] = df[v_cols].std(axis=1)
```

#### `v_skew`
```python
df['v_skew'] = df[v_cols].skew(axis=1)
```

#### `v_kurtosis`
```python
df['v_kurtosis'] = df[v_cols].kurtosis(axis=1)
```

---

## Using the Feature Engineer

### Basic Usage

```python
from src import FraudFeatureEngineer

# Initialize
engineer = FraudFeatureEngineer()

# Fit on training data
X_train_fe = engineer.fit_transform(X_train)

# Transform test data (uses training statistics)
X_test_fe = engineer.transform(X_test)

# View new features
print(f"Original features: {X_train.shape[1]}")
print(f"Engineered features: {X_train_fe.shape[1]}")
```

### Available Configuration

```python
engineer = FraudFeatureEngineer(
    add_amount_features=True,    # Amount-based features
    add_time_features=True,      # Temporal features
    add_v_features=True,         # PCA-based features
    add_interactions=True,       # Feature interactions
    outlier_threshold=3.0        # Z-score threshold for outliers
)
```

---

## Feature Validation

### Statistical Tests

```python
from src import FeatureValidator

validator = FeatureValidator()

# Run validation
results = validator.validate_univariate(X, y)

# View results
for feature, stats in results.items():
    print(f"{feature}:")
    print(f"  p-value: {stats['p_value']:.4f}")
    print(f"  effect_size: {stats['effect_size']:.3f}")
```

### Test Types

| Feature Type | Test | Effect Size |
|--------------|------|-------------|
| Continuous | Mann-Whitney U | Cohen's d |
| Binary | Chi-square | Cramér's V |
| Categorical | Chi-square | Cramér's V |

### Interpretation

| Effect Size | Cohen's d | Cramér's V | Interpretation |
|-------------|-----------|------------|----------------|
| Small | < 0.2 | < 0.1 | Minimal difference |
| Medium | 0.2 - 0.8 | 0.1 - 0.3 | Moderate difference |
| Large | > 0.8 | > 0.3 | Substantial difference |

---

## Feature Selection

### Three-Method Approach

```python
from src import FeatureValidator

validator = FeatureValidator()

# 1. Filter Method (statistical tests)
filter_selected = validator.select_features(
    X, y, method='filter', threshold=0.05
)

# 2. Wrapper Method (recursive elimination)
wrapper_selected = validator.select_features(
    X, y, method='wrapper', n_features=20
)

# 3. Embedded Method (L1 regularization)
embedded_selected = validator.select_features(
    X, y, method='embedded', alpha=0.01
)

# 4. Ensemble (combine all methods)
ensemble_selected = validator.select_features(
    X, y, method='ensemble', min_votes=2
)
```

### Selection Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE SELECTION                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   All Features (50+)                                            │
│         │                                                       │
│         ▼                                                       │
│   ┌─────────────┐                                               │
│   │   FILTER    │  Statistical significance (p < 0.05)          │
│   └──────┬──────┘                                               │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────┐                                               │
│   │   WRAPPER   │  Recursive Feature Elimination                │
│   └──────┬──────┘                                               │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────┐                                               │
│   │  EMBEDDED   │  L1 Regularization                            │
│   └──────┬──────┘                                               │
│          │                                                      │
│          ▼                                                      │
│   Final Features (20-30)                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Feature Importance Results

Top features by importance (typical results):

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | V14 | 0.142 | PCA |
| 2 | V4 | 0.098 | PCA |
| 3 | V12 | 0.087 | PCA |
| 4 | amount_zscore | 0.076 | Engineered |
| 5 | V10 | 0.065 | PCA |
| 6 | v_outlier_count | 0.058 | Engineered |
| 7 | is_night | 0.045 | Engineered |
| 8 | V17 | 0.043 | PCA |
| 9 | amount_log | 0.038 | Engineered |
| 10 | v1_v2_interaction | 0.035 | Engineered |

---

## Best Practices

### Do's
- Validate features statistically before using
- Use domain knowledge to guide feature creation
- Test features on holdout data
- Document feature rationale

### Don'ts
- Don't create features from target variable
- Don't use test data for feature engineering
- Don't ignore feature correlations
- Don't skip validation step

---

## Next Steps

- [[Model Training]] - Use features for training
- [[API Reference]] - Full code documentation
- [[Imbalanced Data Handling]] - Sampling strategies

---

<div align="center">

**Author:** [Iman Elshazli](https://www.linkedin.com/in/monna1478/)

</div>
