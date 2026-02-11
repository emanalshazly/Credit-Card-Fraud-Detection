"""
Feature Validation Module for Credit Card Fraud Detection.

Statistical validation to ensure features are meaningful:
1. Univariate Analysis - t-tests, chi-square, Mann-Whitney
2. Multivariate Analysis - feature importance, permutation importance
3. Correlation Analysis - redundancy detection
4. Feature Selection - filter, wrapper, embedded methods
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import (
    VarianceThreshold, mutual_info_classif, RFE,
    SelectKBest, f_classif
)
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from collections import Counter


class FeatureValidator:
    """
    Comprehensive feature validation for fraud detection.

    Performs:
    1. Univariate statistical tests (fraud vs normal)
    2. Feature importance (Gini, Permutation)
    3. Correlation analysis
    4. Feature selection (Filter, Wrapper, Embedded)
    """

    def __init__(self, significance_level: float = 0.05,
                 correlation_threshold: float = 0.9,
                 importance_threshold: float = 0.01):
        self.significance_level = significance_level
        self.correlation_threshold = correlation_threshold
        self.importance_threshold = importance_threshold

        self.validation_results_ = {}
        self.selected_features_ = []

    def validate_univariate(self, X: pd.DataFrame, y: pd.Series,
                           verbose: bool = True) -> pd.DataFrame:
        """
        Perform univariate statistical tests for each feature.

        Args:
            X: Feature DataFrame
            y: Target series (0=normal, 1=fraud)
            verbose: Print results

        Returns:
            DataFrame with validation results
        """
        results = []

        fraud_mask = (y == 1)
        normal_mask = (y == 0)

        for col in X.columns:
            fraud_values = X.loc[fraud_mask, col].dropna()
            normal_values = X.loc[normal_mask, col].dropna()

            # Determine if binary or continuous
            unique_values = X[col].nunique()
            is_binary = unique_values <= 2

            if is_binary:
                # Chi-square test for categorical features
                contingency = pd.crosstab(X[col], y)
                if contingency.shape == (2, 2):
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                    test_name = "Chi-square"
                    effect_size = np.sqrt(chi2 / len(y))  # Cramér's V for 2x2
                else:
                    p_value = 1.0
                    test_name = "Chi-square (invalid)"
                    effect_size = 0
            else:
                # Mann-Whitney U test (non-parametric, robust to skewness)
                statistic, p_value = stats.mannwhitneyu(
                    fraud_values, normal_values, alternative='two-sided'
                )
                test_name = "Mann-Whitney U"

                # Cohen's d effect size
                pooled_std = np.sqrt(
                    (normal_values.std()**2 + fraud_values.std()**2) / 2
                )
                effect_size = (fraud_values.mean() - normal_values.mean()) / (pooled_std + 1e-8)

            # Interpretation
            if abs(effect_size) < 0.2:
                effect_interpretation = "Negligible"
            elif abs(effect_size) < 0.5:
                effect_interpretation = "Small"
            elif abs(effect_size) < 0.8:
                effect_interpretation = "Medium"
            else:
                effect_interpretation = "Large"

            significant = p_value < self.significance_level

            results.append({
                'feature': col,
                'test': test_name,
                'p_value': p_value,
                'effect_size': effect_size,
                'effect_interpretation': effect_interpretation,
                'significant': significant,
                'normal_mean': normal_values.mean(),
                'fraud_mean': fraud_values.mean(),
                'mean_diff': fraud_values.mean() - normal_values.mean(),
                'decision': '✅ KEEP' if significant and abs(effect_size) > 0.1 else '⚠️ REVIEW' if significant else '❌ DISCARD'
            })

        results_df = pd.DataFrame(results).sort_values('p_value')
        self.validation_results_['univariate'] = results_df

        if verbose:
            print("\n" + "="*80)
            print("UNIVARIATE FEATURE VALIDATION")
            print("="*80)
            print(f"\nTotal features: {len(results_df)}")
            print(f"Significant (p < {self.significance_level}): {results_df['significant'].sum()}")
            print(f"Large effect size: {(results_df['effect_interpretation'] == 'Large').sum()}")
            print("\nTop 10 Features by Effect Size:")
            print(results_df.nlargest(10, 'effect_size')[['feature', 'effect_size', 'p_value', 'decision']])

        return results_df

    def validate_multivariate(self, X: pd.DataFrame, y: pd.Series,
                             verbose: bool = True) -> pd.DataFrame:
        """
        Calculate feature importance using Random Forest.

        Args:
            X: Feature DataFrame
            y: Target series
            verbose: Print results

        Returns:
            DataFrame with importance scores
        """
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        rf.fit(X, y)

        # Gini importance
        gini_importance = pd.DataFrame({
            'feature': X.columns,
            'gini_importance': rf.feature_importances_
        })

        # Permutation importance (more reliable but slower)
        perm_result = permutation_importance(
            rf, X, y, n_repeats=10, random_state=42, n_jobs=-1
        )
        perm_importance = pd.DataFrame({
            'feature': X.columns,
            'perm_importance': perm_result.importances_mean,
            'perm_std': perm_result.importances_std
        })

        # Merge results
        importance_df = gini_importance.merge(perm_importance, on='feature')
        importance_df['avg_importance'] = (
            importance_df['gini_importance'] + importance_df['perm_importance']
        ) / 2
        importance_df = importance_df.sort_values('avg_importance', ascending=False)

        # Decision based on importance threshold
        importance_df['decision'] = importance_df['avg_importance'].apply(
            lambda x: '✅ KEEP' if x > self.importance_threshold else '❌ DISCARD'
        )

        self.validation_results_['multivariate'] = importance_df

        if verbose:
            print("\n" + "="*80)
            print("MULTIVARIATE FEATURE IMPORTANCE")
            print("="*80)
            print(f"\nImportance threshold: {self.importance_threshold}")
            print(f"Features above threshold: {(importance_df['avg_importance'] > self.importance_threshold).sum()}")
            print("\nTop 15 Features:")
            print(importance_df.head(15)[['feature', 'gini_importance', 'perm_importance', 'decision']])

        return importance_df

    def validate_correlation(self, X: pd.DataFrame,
                            verbose: bool = True) -> tuple:
        """
        Analyze feature correlations and detect redundancy.

        Args:
            X: Feature DataFrame
            verbose: Print results

        Returns:
            Tuple of (correlation_matrix, high_correlation_pairs)
        """
        corr_matrix = X.corr().abs()

        # Find highly correlated pairs
        high_corr_pairs = []
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        for col in upper_tri.columns:
            for idx in upper_tri.index:
                if upper_tri.loc[idx, col] > self.correlation_threshold:
                    high_corr_pairs.append({
                        'feature_1': idx,
                        'feature_2': col,
                        'correlation': upper_tri.loc[idx, col]
                    })

        high_corr_df = pd.DataFrame(high_corr_pairs)
        self.validation_results_['correlation'] = {
            'matrix': corr_matrix,
            'high_pairs': high_corr_df
        }

        if verbose:
            print("\n" + "="*80)
            print("CORRELATION ANALYSIS")
            print("="*80)
            print(f"\nCorrelation threshold: {self.correlation_threshold}")
            print(f"Highly correlated pairs: {len(high_corr_pairs)}")
            if len(high_corr_pairs) > 0:
                print("\nRedundant feature pairs:")
                print(high_corr_df)

        return corr_matrix, high_corr_df

    def select_features(self, X: pd.DataFrame, y: pd.Series,
                       method: str = 'ensemble', k: int = 20,
                       verbose: bool = True) -> list:
        """
        Select best features using various methods.

        Args:
            X: Feature DataFrame
            y: Target series
            method: 'filter', 'wrapper', 'embedded', or 'ensemble'
            k: Number of features to select
            verbose: Print results

        Returns:
            List of selected feature names
        """
        selected = {}

        # 1. Filter Method: Mutual Information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_df = pd.DataFrame({
            'feature': X.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        selected['filter_mi'] = set(mi_df.head(k)['feature'].tolist())

        # 2. Filter Method: F-Score (ANOVA)
        selector_f = SelectKBest(f_classif, k=k)
        selector_f.fit(X, y)
        f_mask = selector_f.get_support()
        selected['filter_f'] = set(X.columns[f_mask].tolist())

        # 3. Wrapper Method: RFE with Random Forest
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rfe = RFE(estimator=rf, n_features_to_select=k, step=5)
        rfe.fit(X, y)
        selected['wrapper_rfe'] = set(X.columns[rfe.support_].tolist())

        # 4. Embedded Method: L1 Regularization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lasso = LogisticRegressionCV(
            cv=5, penalty='l1', solver='saga',
            random_state=42, max_iter=5000, n_jobs=-1
        )
        lasso.fit(X_scaled, y)
        lasso_mask = lasso.coef_[0] != 0
        selected['embedded_l1'] = set(X.columns[lasso_mask].tolist())

        # Ensemble: Features selected by 2+ methods
        all_selections = list(selected.values())
        feature_votes = Counter()
        for selection in all_selections:
            feature_votes.update(selection)

        if method == 'ensemble':
            consensus_features = [f for f, votes in feature_votes.items() if votes >= 2]
            self.selected_features_ = consensus_features
        else:
            self.selected_features_ = list(selected.get(f'{method}_mi', selected.get(method, [])))

        self.validation_results_['selection'] = {
            'methods': selected,
            'votes': feature_votes,
            'final': self.selected_features_
        }

        if verbose:
            print("\n" + "="*80)
            print("FEATURE SELECTION RESULTS")
            print("="*80)
            print(f"\nMethod: {method}")
            print(f"Filter (MI): {len(selected['filter_mi'])} features")
            print(f"Filter (F-Score): {len(selected['filter_f'])} features")
            print(f"Wrapper (RFE): {len(selected['wrapper_rfe'])} features")
            print(f"Embedded (L1): {len(selected['embedded_l1'])} features")
            print(f"\nFinal selected: {len(self.selected_features_)} features")

            print("\nTop features by vote count:")
            for feature, votes in feature_votes.most_common(15):
                print(f"  {feature}: {votes}/4 methods")

        return self.selected_features_

    def plot_validation_summary(self, X: pd.DataFrame, y: pd.Series,
                               figsize: tuple = (16, 12)):
        """
        Create comprehensive validation visualization.
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Feature Importance Bar Plot
        if 'multivariate' in self.validation_results_:
            imp_df = self.validation_results_['multivariate'].head(20)
            axes[0, 0].barh(imp_df['feature'], imp_df['avg_importance'], color='steelblue')
            axes[0, 0].set_xlabel('Average Importance')
            axes[0, 0].set_title('Top 20 Features by Importance')
            axes[0, 0].invert_yaxis()

        # 2. Effect Size Distribution
        if 'univariate' in self.validation_results_:
            uni_df = self.validation_results_['univariate']
            colors = ['green' if d == '✅ KEEP' else 'red' for d in uni_df['decision']]
            axes[0, 1].scatter(uni_df['effect_size'], -np.log10(uni_df['p_value']),
                              c=colors, alpha=0.6)
            axes[0, 1].axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
            axes[0, 1].axvline(x=0, color='gray', linestyle='-', alpha=0.5)
            axes[0, 1].set_xlabel('Effect Size (Cohen\'s d)')
            axes[0, 1].set_ylabel('-log10(p-value)')
            axes[0, 1].set_title('Volcano Plot: Effect Size vs Significance')
            axes[0, 1].legend()

        # 3. Correlation Heatmap (top features)
        if 'multivariate' in self.validation_results_:
            top_features = self.validation_results_['multivariate'].head(15)['feature'].tolist()
            corr = X[top_features].corr()
            sns.heatmap(corr, ax=axes[1, 0], cmap='coolwarm', center=0,
                       annot=True, fmt='.2f', square=True)
            axes[1, 0].set_title('Correlation Matrix (Top 15 Features)')

        # 4. Class Distribution by Top Features
        if 'multivariate' in self.validation_results_:
            top_feature = self.validation_results_['multivariate'].iloc[0]['feature']
            fraud = X.loc[y == 1, top_feature]
            normal = X.loc[y == 0, top_feature]
            axes[1, 1].hist(normal, bins=50, alpha=0.7, label='Normal', color='green', density=True)
            axes[1, 1].hist(fraud, bins=50, alpha=0.7, label='Fraud', color='red', density=True)
            axes[1, 1].set_xlabel(top_feature)
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].set_title(f'Distribution: {top_feature}')
            axes[1, 1].legend()

        plt.tight_layout()
        return fig

    def get_validation_summary(self) -> dict:
        """
        Get summary of all validation results.

        Returns:
            Dictionary with validation summary
        """
        summary = {
            'total_features_analyzed': 0,
            'significant_features': 0,
            'high_importance_features': 0,
            'redundant_pairs': 0,
            'final_selected': len(self.selected_features_)
        }

        if 'univariate' in self.validation_results_:
            uni = self.validation_results_['univariate']
            summary['total_features_analyzed'] = len(uni)
            summary['significant_features'] = uni['significant'].sum()

        if 'multivariate' in self.validation_results_:
            multi = self.validation_results_['multivariate']
            summary['high_importance_features'] = (
                multi['avg_importance'] > self.importance_threshold
            ).sum()

        if 'correlation' in self.validation_results_:
            summary['redundant_pairs'] = len(
                self.validation_results_['correlation']['high_pairs']
            )

        return summary


def quick_feature_validation(X: pd.DataFrame, y: pd.Series, k: int = 25) -> list:
    """
    Quick feature validation and selection.

    Args:
        X: Features DataFrame
        y: Target series
        k: Number of features to select

    Returns:
        List of selected feature names
    """
    validator = FeatureValidator()

    # Run all validations
    validator.validate_univariate(X, y, verbose=True)
    validator.validate_multivariate(X, y, verbose=True)
    validator.validate_correlation(X, verbose=True)
    selected = validator.select_features(X, y, method='ensemble', k=k, verbose=True)

    return selected
