"""
Feature Engineering for Credit Card Fraud Detection.

Domain-driven feature creation based on fraud patterns:
1. Amount Anomalies - unusual transaction amounts
2. Time Patterns - late night, velocity
3. Statistical Deviations - z-scores, percentiles
4. PCA Component Interactions - V feature combinations
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Production-ready Feature Engineering Pipeline for Fraud Detection.

    Implements domain-driven features based on fraud patterns:
    - Amount-based: z-scores, round numbers, log transform
    - Time-based: hour of day, is_night, time bins
    - Statistical: deviations, percentiles
    - PCA interactions: top feature combinations

    Usage:
        engineer = FraudFeatureEngineer()
        X_train_engineered = engineer.fit_transform(X_train)
        X_test_engineered = engineer.transform(X_test)  # Only transform!
    """

    def __init__(self, create_interactions: bool = True,
                 create_time_features: bool = True,
                 create_amount_features: bool = True,
                 create_statistical_features: bool = True):
        self.create_interactions = create_interactions
        self.create_time_features = create_time_features
        self.create_amount_features = create_amount_features
        self.create_statistical_features = create_statistical_features

        # Will be set during fit
        self.amount_stats_ = None
        self.v_feature_stats_ = None
        self.fitted_ = False

    def fit(self, X, y=None):
        """
        Learn statistics from training data.

        Args:
            X: DataFrame with columns [Time, Amount, V1-V28]
            y: Target (optional, not used)

        Returns:
            self
        """
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        # Amount statistics (for z-score calculation)
        if 'Amount' in X.columns:
            self.amount_stats_ = {
                'mean': X['Amount'].mean(),
                'std': X['Amount'].std(),
                'median': X['Amount'].median(),
                'q25': X['Amount'].quantile(0.25),
                'q75': X['Amount'].quantile(0.75),
                'q95': X['Amount'].quantile(0.95),
                'q99': X['Amount'].quantile(0.99)
            }

        # V-feature statistics (for PCA component engineering)
        v_cols = [col for col in X.columns if col.startswith('V')]
        if v_cols:
            self.v_feature_stats_ = {}
            for col in v_cols:
                self.v_feature_stats_[col] = {
                    'mean': X[col].mean(),
                    'std': X[col].std()
                }

        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Create engineered features.

        Args:
            X: DataFrame with columns [Time, Amount, V1-V28]

        Returns:
            DataFrame with engineered features
        """
        if not self.fitted_:
            raise ValueError("Must call fit() before transform()")

        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        # === AMOUNT-BASED FEATURES ===
        if self.create_amount_features and 'Amount' in X.columns:
            X = self._create_amount_features(X)

        # === TIME-BASED FEATURES ===
        if self.create_time_features and 'Time' in X.columns:
            X = self._create_time_features(X)

        # === STATISTICAL FEATURES ===
        if self.create_statistical_features:
            X = self._create_statistical_features(X)

        # === PCA INTERACTION FEATURES ===
        if self.create_interactions:
            X = self._create_pca_interactions(X)

        return X

    def _create_amount_features(self, X):
        """Create amount-based features."""

        # 1. Amount Z-Score (how unusual is this amount?)
        X['amount_zscore'] = (X['Amount'] - self.amount_stats_['mean']) / (self.amount_stats_['std'] + 1e-8)

        # 2. Amount Log Transform (handle skewness)
        X['amount_log'] = np.log1p(X['Amount'])

        # 3. Amount Round Flag (fraudsters use round numbers: $50, $100, $500)
        X['amount_is_round'] = ((X['Amount'] % 10 == 0) |
                                (X['Amount'] % 50 == 0) |
                                (X['Amount'] % 100 == 0)).astype(int)

        # 4. Amount Percentile Bins
        X['amount_is_high'] = (X['Amount'] > self.amount_stats_['q95']).astype(int)
        X['amount_is_very_high'] = (X['Amount'] > self.amount_stats_['q99']).astype(int)

        # 5. Amount vs Median Ratio
        X['amount_vs_median'] = X['Amount'] / (self.amount_stats_['median'] + 1)

        # 6. Amount Bins (categorical encoding)
        X['amount_bin'] = pd.cut(X['Amount'],
                                  bins=[0, 10, 50, 100, 500, 1000, 5000, float('inf')],
                                  labels=[0, 1, 2, 3, 4, 5, 6]).astype(int)

        # 7. Small Amount Flag (card testing pattern)
        X['amount_is_small'] = (X['Amount'] < 10).astype(int)

        return X

    def _create_time_features(self, X):
        """Create time-based features."""

        # Convert seconds to hours (assuming Time is seconds since first transaction)
        seconds_in_day = 86400

        # 1. Hour of Day (0-23)
        X['hour_of_day'] = ((X['Time'] % seconds_in_day) / 3600).astype(int)

        # 2. Is Night (12am - 6am) - Fraud often happens when victim is asleep
        X['is_night'] = ((X['hour_of_day'] >= 0) & (X['hour_of_day'] <= 6)).astype(int)

        # 3. Is Business Hours (9am - 5pm)
        X['is_business_hours'] = ((X['hour_of_day'] >= 9) & (X['hour_of_day'] <= 17)).astype(int)

        # 4. Time Bins (morning, afternoon, evening, night)
        X['time_bin'] = pd.cut(X['hour_of_day'],
                               bins=[-1, 6, 12, 18, 24],
                               labels=[0, 1, 2, 3]).astype(int)

        # 5. Day Number (which day in the dataset)
        X['day_number'] = (X['Time'] // seconds_in_day).astype(int)

        # 6. Is Day 1 or Day 2
        X['is_day_1'] = (X['day_number'] == 0).astype(int)

        return X

    def _create_statistical_features(self, X):
        """Create statistical deviation features."""

        v_cols = [col for col in X.columns if col.startswith('V') and col[1:].isdigit()]

        if not v_cols:
            return X

        # 1. Sum of absolute V values (overall anomaly score)
        X['v_sum_abs'] = X[v_cols].abs().sum(axis=1)

        # 2. Mean of V values
        X['v_mean'] = X[v_cols].mean(axis=1)

        # 3. Std of V values (variability)
        X['v_std'] = X[v_cols].std(axis=1)

        # 4. Max absolute V value
        X['v_max_abs'] = X[v_cols].abs().max(axis=1)

        # 5. Count of V values > 2 std (outlier count)
        X['v_outlier_count'] = (X[v_cols].abs() > 2).sum(axis=1)

        # 6. Skewness of V values
        X['v_skew'] = X[v_cols].apply(lambda row: stats.skew(row), axis=1)

        # 7. Kurtosis of V values
        X['v_kurtosis'] = X[v_cols].apply(lambda row: stats.kurtosis(row), axis=1)

        return X

    def _create_pca_interactions(self, X):
        """Create PCA component interaction features."""

        # Based on typical fraud detection, V14, V12, V10, V17, V4 are often important
        important_v_cols = ['V14', 'V12', 'V10', 'V17', 'V4', 'V11', 'V3', 'V16']
        available_cols = [col for col in important_v_cols if col in X.columns]

        if len(available_cols) < 2:
            return X

        # 1. Key squared features (non-linear relationships)
        if 'V14' in X.columns:
            X['V14_squared'] = X['V14'] ** 2
        if 'V17' in X.columns:
            X['V17_squared'] = X['V17'] ** 2
        if 'V12' in X.columns:
            X['V12_squared'] = X['V12'] ** 2

        # 2. Key interactions
        if 'V14' in X.columns and 'V12' in X.columns:
            X['V14_x_V12'] = X['V14'] * X['V12']
        if 'V10' in X.columns and 'V17' in X.columns:
            X['V10_x_V17'] = X['V10'] * X['V17']
        if 'V4' in X.columns and 'V11' in X.columns:
            X['V4_x_V11'] = X['V4'] * X['V11']
        if 'V3' in X.columns and 'V16' in X.columns:
            X['V3_x_V16'] = X['V3'] * X['V16']

        # 3. Amount interactions with important V features
        if 'Amount' in X.columns:
            if 'V14' in X.columns:
                X['Amount_x_V14'] = X['Amount'] * X['V14']
            if 'V17' in X.columns:
                X['Amount_x_V17'] = X['Amount'] * X['V17']

        return X

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_feature_names(self):
        """Return list of all engineered feature names."""
        features = []

        if self.create_amount_features:
            features.extend([
                'amount_zscore', 'amount_log', 'amount_is_round',
                'amount_is_high', 'amount_is_very_high', 'amount_vs_median',
                'amount_bin', 'amount_is_small'
            ])

        if self.create_time_features:
            features.extend([
                'hour_of_day', 'is_night', 'is_business_hours',
                'time_bin', 'day_number', 'is_day_1'
            ])

        if self.create_statistical_features:
            features.extend([
                'v_sum_abs', 'v_mean', 'v_std', 'v_max_abs',
                'v_outlier_count', 'v_skew', 'v_kurtosis'
            ])

        if self.create_interactions:
            features.extend([
                'V14_squared', 'V17_squared', 'V12_squared',
                'V14_x_V12', 'V10_x_V17', 'V4_x_V11', 'V3_x_V16',
                'Amount_x_V14', 'Amount_x_V17'
            ])

        return features


def create_risk_score(df: pd.DataFrame) -> pd.Series:
    """
    Create interpretable composite risk score.

    Business Logic:
    - Multiple weak signals combine to strong fraud indicator
    - Weighted based on fraud detection domain knowledge

    Args:
        df: DataFrame with engineered features

    Returns:
        Series with risk scores (0-15 range)
    """
    score = pd.Series(0, index=df.index)

    # Amount anomalies (+4 max)
    if 'amount_zscore' in df.columns:
        score += (df['amount_zscore'].abs() > 2).astype(int) * 2
        score += (df['amount_zscore'].abs() > 3).astype(int) * 2

    # Round amount flag (+1)
    if 'amount_is_round' in df.columns:
        score += df['amount_is_round']

    # Small amount (card testing) (+2)
    if 'amount_is_small' in df.columns:
        score += df['amount_is_small'] * 2

    # Night transaction (+2)
    if 'is_night' in df.columns:
        score += df['is_night'] * 2

    # High V outlier count (+3)
    if 'v_outlier_count' in df.columns:
        score += (df['v_outlier_count'] > 5).astype(int) * 3

    # Very high amount (+3)
    if 'amount_is_very_high' in df.columns:
        score += df['amount_is_very_high'] * 3

    return score


def get_feature_importance_by_domain(feature_names: list) -> dict:
    """
    Categorize features by domain for analysis.

    Args:
        feature_names: List of feature names

    Returns:
        Dictionary with categorized features
    """
    categories = {
        'amount': [],
        'time': [],
        'pca_original': [],
        'pca_engineered': [],
        'statistical': [],
        'interactions': []
    }

    for feature in feature_names:
        if feature.startswith('Amount') or feature.startswith('amount'):
            categories['amount'].append(feature)
        elif feature in ['Time', 'hour_of_day', 'is_night', 'is_business_hours',
                        'time_bin', 'day_number', 'is_day_1']:
            categories['time'].append(feature)
        elif feature.startswith('V') and '_' not in feature:
            categories['pca_original'].append(feature)
        elif 'squared' in feature or '_x_' in feature:
            categories['interactions'].append(feature)
        elif feature.startswith('v_'):
            categories['statistical'].append(feature)
        elif feature.startswith('V'):
            categories['pca_engineered'].append(feature)

    return categories
