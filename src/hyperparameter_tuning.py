"""
Hyperparameter Tuning for Credit Card Fraud Detection.

Methods:
1. GridSearchCV - Exhaustive search
2. RandomizedSearchCV - Random sampling
3. Optuna - Bayesian optimization
4. Cross-validation with Stratified K-Fold
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score

# Try to import optuna, but don't fail if not installed
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# Custom scorers for fraud detection (focus on minority class)
f1_fraud = make_scorer(f1_score, pos_label=1)
precision_fraud = make_scorer(precision_score, pos_label=1, zero_division=0)
recall_fraud = make_scorer(recall_score, pos_label=1, zero_division=0)


def get_param_grids() -> Dict[str, Dict[str, Any]]:
    """
    Get predefined hyperparameter grids for each model.

    Returns:
        Dictionary with parameter grids
    """
    return {
        'logistic_regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['saga'],
            'max_iter': [1000],
            'class_weight': ['balanced', None]
        },
        'random_forest': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample', None]
        },
        'xgboost': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2]
        }
    }


def get_param_distributions() -> Dict[str, Dict[str, Any]]:
    """
    Get parameter distributions for RandomizedSearchCV.

    Returns:
        Dictionary with parameter distributions
    """
    from scipy.stats import uniform, randint, loguniform

    return {
        'logistic_regression': {
            'C': loguniform(1e-4, 1e2),
            'penalty': ['l1', 'l2'],
            'solver': ['saga'],
            'max_iter': [1000],
            'class_weight': ['balanced', None]
        },
        'random_forest': {
            'n_estimators': randint(50, 500),
            'max_depth': randint(3, 30),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'class_weight': ['balanced', 'balanced_subsample']
        },
        'xgboost': {
            'n_estimators': randint(50, 500),
            'max_depth': randint(3, 15),
            'learning_rate': loguniform(0.01, 0.3),
            'subsample': uniform(0.5, 0.5),
            'colsample_bytree': uniform(0.5, 0.5),
            'min_child_weight': randint(1, 10),
            'gamma': uniform(0, 0.5)
        }
    }


class HyperparameterTuner:
    """
    Comprehensive hyperparameter tuning for fraud detection models.

    Supports:
    - GridSearchCV (exhaustive)
    - RandomizedSearchCV (faster)
    - Optuna (Bayesian optimization)
    """

    def __init__(self, cv: int = 5, scoring: str = 'f1',
                 n_jobs: int = -1, random_state: int = 42):
        """
        Initialize tuner.

        Args:
            cv: Number of cross-validation folds
            scoring: Scoring metric ('f1', 'precision', 'recall', 'roc_auc')
            n_jobs: Number of parallel jobs
            random_state: Random seed
        """
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Set up scorer
        if scoring == 'f1':
            self.scorer = f1_fraud
        elif scoring == 'precision':
            self.scorer = precision_fraud
        elif scoring == 'recall':
            self.scorer = recall_fraud
        else:
            self.scorer = scoring

        self.scoring_name = scoring

        # Results storage
        self.best_params_ = {}
        self.best_scores_ = {}
        self.cv_results_ = {}

    def grid_search(self, X, y, model_name: str = 'xgboost',
                   verbose: int = 1) -> Tuple[Any, Dict]:
        """
        Perform GridSearchCV.

        Args:
            X: Features
            y: Target
            model_name: 'logistic_regression', 'random_forest', or 'xgboost'
            verbose: Verbosity level

        Returns:
            Tuple of (best_model, best_params)
        """
        param_grid = get_param_grids()[model_name]
        model = self._get_base_model(model_name)

        cv_splitter = StratifiedKFold(n_splits=self.cv, shuffle=True,
                                      random_state=self.random_state)

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=self.scorer,
            cv=cv_splitter,
            n_jobs=self.n_jobs,
            verbose=verbose,
            refit=True
        )

        grid_search.fit(X, y)

        self.best_params_[model_name] = grid_search.best_params_
        self.best_scores_[model_name] = grid_search.best_score_
        self.cv_results_[model_name] = pd.DataFrame(grid_search.cv_results_)

        print(f"\n{'='*60}")
        print(f"GridSearchCV Results - {model_name}")
        print(f"{'='*60}")
        print(f"Best {self.scoring_name}: {grid_search.best_score_:.4f}")
        print(f"Best Parameters: {grid_search.best_params_}")

        return grid_search.best_estimator_, grid_search.best_params_

    def random_search(self, X, y, model_name: str = 'xgboost',
                     n_iter: int = 50, verbose: int = 1) -> Tuple[Any, Dict]:
        """
        Perform RandomizedSearchCV.

        Args:
            X: Features
            y: Target
            model_name: Model name
            n_iter: Number of random combinations to try
            verbose: Verbosity level

        Returns:
            Tuple of (best_model, best_params)
        """
        param_dist = get_param_distributions()[model_name]
        model = self._get_base_model(model_name)

        cv_splitter = StratifiedKFold(n_splits=self.cv, shuffle=True,
                                      random_state=self.random_state)

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=self.scorer,
            cv=cv_splitter,
            n_jobs=self.n_jobs,
            verbose=verbose,
            random_state=self.random_state,
            refit=True
        )

        random_search.fit(X, y)

        self.best_params_[model_name] = random_search.best_params_
        self.best_scores_[model_name] = random_search.best_score_
        self.cv_results_[model_name] = pd.DataFrame(random_search.cv_results_)

        print(f"\n{'='*60}")
        print(f"RandomizedSearchCV Results - {model_name}")
        print(f"{'='*60}")
        print(f"Best {self.scoring_name}: {random_search.best_score_:.4f}")
        print(f"Best Parameters: {random_search.best_params_}")

        return random_search.best_estimator_, random_search.best_params_

    def optuna_search(self, X, y, model_name: str = 'xgboost',
                     n_trials: int = 100, verbose: bool = True) -> Tuple[Any, Dict]:
        """
        Perform Optuna Bayesian optimization.

        Args:
            X: Features
            y: Target
            model_name: Model name
            n_trials: Number of optimization trials
            verbose: Show progress

        Returns:
            Tuple of (best_model, best_params)
        """
        if not OPTUNA_AVAILABLE:
            print("Optuna not installed. Falling back to RandomizedSearchCV.")
            return self.random_search(X, y, model_name, n_iter=n_trials)

        def objective(trial):
            if model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'random_state': self.random_state,
                    'use_label_encoder': False,
                    'eval_metric': 'logloss',
                    'n_jobs': -1
                }
                # Calculate scale_pos_weight
                neg_count = np.sum(y == 0)
                pos_count = np.sum(y == 1)
                params['scale_pos_weight'] = neg_count / pos_count

                model = XGBClassifier(**params)

            elif model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample']),
                    'random_state': self.random_state,
                    'n_jobs': -1
                }
                model = RandomForestClassifier(**params)

            elif model_name == 'logistic_regression':
                params = {
                    'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                    'solver': 'saga',
                    'max_iter': 1000,
                    'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                    'random_state': self.random_state,
                    'n_jobs': -1
                }
                model = LogisticRegression(**params)

            # Cross-validation
            cv_splitter = StratifiedKFold(n_splits=self.cv, shuffle=True,
                                          random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=self.scorer, n_jobs=1)
            return scores.mean()

        # Create study
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='maximize', sampler=sampler)

        # Optimize
        verbosity = optuna.logging.INFO if verbose else optuna.logging.WARNING
        optuna.logging.set_verbosity(verbosity)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

        # Get best parameters and train final model
        best_params = study.best_params
        self.best_params_[model_name] = best_params
        self.best_scores_[model_name] = study.best_value

        # Train final model with best params
        final_model = self._get_model_with_params(model_name, best_params, y)
        final_model.fit(X, y)

        print(f"\n{'='*60}")
        print(f"Optuna Optimization Results - {model_name}")
        print(f"{'='*60}")
        print(f"Best {self.scoring_name}: {study.best_value:.4f}")
        print(f"Best Parameters: {best_params}")

        return final_model, best_params

    def _get_base_model(self, model_name: str):
        """Get base model instance."""
        if model_name == 'logistic_regression':
            return LogisticRegression(random_state=self.random_state, n_jobs=-1)
        elif model_name == 'random_forest':
            return RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        elif model_name == 'xgboost':
            return XGBClassifier(random_state=self.random_state,
                               use_label_encoder=False,
                               eval_metric='logloss', n_jobs=-1)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _get_model_with_params(self, model_name: str, params: Dict, y) -> Any:
        """Get model with specific parameters."""
        if model_name == 'xgboost':
            neg_count = np.sum(y == 0)
            pos_count = np.sum(y == 1)
            params['scale_pos_weight'] = neg_count / pos_count
            params['random_state'] = self.random_state
            params['use_label_encoder'] = False
            params['eval_metric'] = 'logloss'
            params['n_jobs'] = -1
            return XGBClassifier(**params)

        elif model_name == 'random_forest':
            params['random_state'] = self.random_state
            params['n_jobs'] = -1
            return RandomForestClassifier(**params)

        elif model_name == 'logistic_regression':
            params['solver'] = 'saga'
            params['max_iter'] = 1000
            params['random_state'] = self.random_state
            params['n_jobs'] = -1
            return LogisticRegression(**params)

    def compare_tuned_models(self, X, y, method: str = 'random',
                            n_iter: int = 50) -> pd.DataFrame:
        """
        Tune and compare all models.

        Args:
            X: Features
            y: Target
            method: 'grid', 'random', or 'optuna'
            n_iter: Number of iterations for random/optuna

        Returns:
            DataFrame with comparison results
        """
        results = []

        for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
            print(f"\n{'#'*60}")
            print(f"Tuning: {model_name}")
            print(f"{'#'*60}")

            if method == 'grid':
                best_model, best_params = self.grid_search(X, y, model_name)
            elif method == 'random':
                best_model, best_params = self.random_search(X, y, model_name, n_iter)
            elif method == 'optuna':
                best_model, best_params = self.optuna_search(X, y, model_name, n_iter)

            results.append({
                'model': model_name,
                f'best_{self.scoring_name}': self.best_scores_[model_name],
                'best_params': str(best_params)
            })

        return pd.DataFrame(results).sort_values(f'best_{self.scoring_name}', ascending=False)


def cross_validate_model(model, X, y, cv: int = 5,
                        scoring: list = None) -> pd.DataFrame:
    """
    Perform stratified cross-validation with multiple metrics.

    Args:
        model: Sklearn-compatible model
        X: Features
        y: Target
        cv: Number of folds
        scoring: List of scoring metrics

    Returns:
        DataFrame with CV results
    """
    if scoring is None:
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    results = {}
    for metric in scoring:
        if metric in ['precision', 'recall', 'f1']:
            scorer = make_scorer(eval(f'{metric}_score'), pos_label=1, zero_division=0)
        else:
            scorer = metric

        scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scorer, n_jobs=-1)
        results[metric] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }

    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'metric': list(results.keys()),
        'mean': [results[m]['mean'] for m in results],
        'std': [results[m]['std'] for m in results]
    })

    print("\n" + "="*50)
    print("STRATIFIED K-FOLD CROSS-VALIDATION RESULTS")
    print("="*50)
    print(f"Folds: {cv}")
    print(summary_df.to_string(index=False))

    return summary_df


def quick_tune(X, y, model_name: str = 'xgboost', n_iter: int = 30) -> Tuple[Any, Dict]:
    """
    Quick hyperparameter tuning.

    Args:
        X: Features
        y: Target
        model_name: Model to tune
        n_iter: Number of iterations

    Returns:
        Tuple of (tuned_model, best_params)
    """
    tuner = HyperparameterTuner(cv=5, scoring='f1')
    return tuner.random_search(X, y, model_name, n_iter)
