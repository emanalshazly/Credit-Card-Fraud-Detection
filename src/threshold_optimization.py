"""
Threshold Optimization for Credit Card Fraud Detection.

Optimize classification threshold based on:
1. Business Costs (false positive vs false negative costs)
2. Maximize specific metrics (F1, F-beta, Youden's J)
3. Precision-Recall trade-off analysis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, roc_curve, f1_score,
    precision_score, recall_score, confusion_matrix,
    fbeta_score
)
from typing import Tuple, Dict


class ThresholdOptimizer:
    """
    Optimize classification threshold for fraud detection.

    Methods:
    1. Cost-based optimization (business costs)
    2. F1 maximization
    3. F-beta optimization (custom precision/recall trade-off)
    4. Youden's J statistic (ROC-based)
    5. Precision at minimum recall
    """

    def __init__(self, y_true: np.ndarray, y_prob: np.ndarray):
        """
        Initialize optimizer.

        Args:
            y_true: True labels (0/1)
            y_prob: Predicted probabilities for positive class
        """
        self.y_true = np.array(y_true)
        self.y_prob = np.array(y_prob)

        # Precompute curves
        self.precision, self.recall, self.pr_thresholds = precision_recall_curve(
            y_true, y_prob
        )
        self.fpr, self.tpr, self.roc_thresholds = roc_curve(y_true, y_prob)

        self.optimal_thresholds_ = {}

    def optimize_f1(self) -> Tuple[float, float]:
        """
        Find threshold that maximizes F1 score.

        Returns:
            Tuple of (optimal_threshold, max_f1_score)
        """
        f1_scores = 2 * (self.precision * self.recall) / (self.precision + self.recall + 1e-8)
        optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last element
        optimal_threshold = self.pr_thresholds[optimal_idx]
        max_f1 = f1_scores[optimal_idx]

        self.optimal_thresholds_['f1'] = {
            'threshold': optimal_threshold,
            'score': max_f1,
            'precision': self.precision[optimal_idx],
            'recall': self.recall[optimal_idx]
        }

        print(f"F1 Optimization:")
        print(f"  Optimal Threshold: {optimal_threshold:.4f}")
        print(f"  Max F1 Score: {max_f1:.4f}")
        print(f"  Precision: {self.precision[optimal_idx]:.4f}")
        print(f"  Recall: {self.recall[optimal_idx]:.4f}")

        return optimal_threshold, max_f1

    def optimize_fbeta(self, beta: float = 2.0) -> Tuple[float, float]:
        """
        Find threshold that maximizes F-beta score.

        F-beta weighs recall higher when beta > 1.
        For fraud detection, beta=2 prioritizes catching fraud.

        Args:
            beta: Weight of recall vs precision

        Returns:
            Tuple of (optimal_threshold, max_fbeta_score)
        """
        fbeta_scores = (1 + beta**2) * (self.precision * self.recall) / \
                      (beta**2 * self.precision + self.recall + 1e-8)
        optimal_idx = np.argmax(fbeta_scores[:-1])
        optimal_threshold = self.pr_thresholds[optimal_idx]
        max_fbeta = fbeta_scores[optimal_idx]

        self.optimal_thresholds_[f'f{beta}'] = {
            'threshold': optimal_threshold,
            'score': max_fbeta,
            'precision': self.precision[optimal_idx],
            'recall': self.recall[optimal_idx]
        }

        print(f"\nF{beta} Optimization (recall-focused):")
        print(f"  Optimal Threshold: {optimal_threshold:.4f}")
        print(f"  Max F{beta} Score: {max_fbeta:.4f}")
        print(f"  Precision: {self.precision[optimal_idx]:.4f}")
        print(f"  Recall: {self.recall[optimal_idx]:.4f}")

        return optimal_threshold, max_fbeta

    def optimize_youden_j(self) -> Tuple[float, float]:
        """
        Find threshold that maximizes Youden's J statistic.

        Youden's J = TPR - FPR = Sensitivity + Specificity - 1

        Returns:
            Tuple of (optimal_threshold, max_j)
        """
        j_scores = self.tpr - self.fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = self.roc_thresholds[optimal_idx]
        max_j = j_scores[optimal_idx]

        self.optimal_thresholds_['youden_j'] = {
            'threshold': optimal_threshold,
            'score': max_j,
            'tpr': self.tpr[optimal_idx],
            'fpr': self.fpr[optimal_idx]
        }

        print(f"\nYouden's J Optimization:")
        print(f"  Optimal Threshold: {optimal_threshold:.4f}")
        print(f"  Max J Statistic: {max_j:.4f}")
        print(f"  TPR (Recall): {self.tpr[optimal_idx]:.4f}")
        print(f"  FPR: {self.fpr[optimal_idx]:.4f}")

        return optimal_threshold, max_j

    def optimize_cost(self, cost_fp: float = 10, cost_fn: float = 500,
                     cost_tp: float = 0, cost_tn: float = 0) -> Tuple[float, float]:
        """
        Find threshold that minimizes total business cost.

        For fraud detection:
        - FP cost: Wasted review time (~$10)
        - FN cost: Missed fraud (~$500 average fraud amount)

        Args:
            cost_fp: Cost of false positive (legitimate flagged as fraud)
            cost_fn: Cost of false negative (fraud passes through)
            cost_tp: Cost of true positive (usually 0, fraud caught)
            cost_tn: Cost of true negative (usually 0, legitimate approved)

        Returns:
            Tuple of (optimal_threshold, min_cost)
        """
        thresholds = np.arange(0.01, 1.0, 0.01)
        costs = []

        for thresh in thresholds:
            y_pred = (self.y_prob >= thresh).astype(int)
            cm = confusion_matrix(self.y_true, y_pred)

            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                continue

            total_cost = (cost_fp * fp + cost_fn * fn +
                         cost_tp * tp + cost_tn * tn)
            costs.append(total_cost)

        costs = np.array(costs)
        optimal_idx = np.argmin(costs)
        optimal_threshold = thresholds[optimal_idx]
        min_cost = costs[optimal_idx]

        # Get metrics at optimal threshold
        y_pred_optimal = (self.y_prob >= optimal_threshold).astype(int)
        cm_optimal = confusion_matrix(self.y_true, y_pred_optimal)
        tn, fp, fn, tp = cm_optimal.ravel()

        self.optimal_thresholds_['cost'] = {
            'threshold': optimal_threshold,
            'cost': min_cost,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'tn': tn,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
        }

        print(f"\nCost Optimization:")
        print(f"  Cost FP: ${cost_fp}, Cost FN: ${cost_fn}")
        print(f"  Optimal Threshold: {optimal_threshold:.4f}")
        print(f"  Minimum Total Cost: ${min_cost:,.2f}")
        print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(f"  Precision: {self.optimal_thresholds_['cost']['precision']:.4f}")
        print(f"  Recall: {self.optimal_thresholds_['cost']['recall']:.4f}")

        return optimal_threshold, min_cost

    def optimize_min_recall(self, min_recall: float = 0.80) -> Tuple[float, float]:
        """
        Find threshold with highest precision at minimum recall constraint.

        Args:
            min_recall: Minimum required recall

        Returns:
            Tuple of (optimal_threshold, precision_at_threshold)
        """
        valid_idx = self.recall >= min_recall
        if not valid_idx.any():
            print(f"Warning: Cannot achieve recall >= {min_recall}")
            return None, None

        # Find highest precision among valid thresholds
        valid_precision = self.precision[valid_idx]
        valid_thresholds = np.concatenate([self.pr_thresholds, [1.0]])[valid_idx]
        valid_recall = self.recall[valid_idx]

        optimal_idx = np.argmax(valid_precision)
        optimal_threshold = valid_thresholds[optimal_idx]
        max_precision = valid_precision[optimal_idx]

        self.optimal_thresholds_['min_recall'] = {
            'threshold': optimal_threshold,
            'precision': max_precision,
            'recall': valid_recall[optimal_idx],
            'min_recall_constraint': min_recall
        }

        print(f"\nMin Recall Constraint Optimization:")
        print(f"  Required Recall: >= {min_recall:.2%}")
        print(f"  Optimal Threshold: {optimal_threshold:.4f}")
        print(f"  Precision: {max_precision:.4f}")
        print(f"  Actual Recall: {valid_recall[optimal_idx]:.4f}")

        return optimal_threshold, max_precision

    def get_all_optimal_thresholds(self) -> pd.DataFrame:
        """
        Run all optimization methods and return summary.

        Returns:
            DataFrame with all optimal thresholds
        """
        self.optimize_f1()
        self.optimize_fbeta(beta=2.0)
        self.optimize_youden_j()
        self.optimize_cost()
        self.optimize_min_recall(min_recall=0.80)

        summary = []
        for method, result in self.optimal_thresholds_.items():
            summary.append({
                'method': method,
                'threshold': result.get('threshold', None),
                'precision': result.get('precision', None),
                'recall': result.get('recall', result.get('tpr', None)),
                'score': result.get('score', result.get('cost', None))
            })

        return pd.DataFrame(summary)

    def plot_threshold_analysis(self, figsize: tuple = (16, 12)):
        """
        Comprehensive visualization of threshold analysis.
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Precision-Recall vs Threshold
        ax1 = axes[0, 0]
        ax1.plot(self.pr_thresholds, self.precision[:-1], 'b-', label='Precision')
        ax1.plot(self.pr_thresholds, self.recall[:-1], 'r-', label='Recall')

        # F1 score
        f1 = 2 * (self.precision[:-1] * self.recall[:-1]) / \
             (self.precision[:-1] + self.recall[:-1] + 1e-8)
        ax1.plot(self.pr_thresholds, f1, 'g--', label='F1 Score')

        # Mark optimal F1 threshold
        if 'f1' in self.optimal_thresholds_:
            opt_thresh = self.optimal_thresholds_['f1']['threshold']
            ax1.axvline(x=opt_thresh, color='green', linestyle=':', alpha=0.7,
                       label=f'Optimal F1 ({opt_thresh:.3f})')

        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title('Precision, Recall, F1 vs Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. ROC Curve with Youden's J
        ax2 = axes[0, 1]
        ax2.plot(self.fpr, self.tpr, 'b-', linewidth=2)
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)

        # Mark Youden's J optimal point
        if 'youden_j' in self.optimal_thresholds_:
            opt_fpr = self.optimal_thresholds_['youden_j']['fpr']
            opt_tpr = self.optimal_thresholds_['youden_j']['tpr']
            ax2.scatter([opt_fpr], [opt_tpr], color='red', s=100, zorder=5,
                       label=f"Youden's J Optimal\nThresh={self.optimal_thresholds_['youden_j']['threshold']:.3f}")

        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title("ROC Curve with Youden's J Optimal Point")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Cost vs Threshold
        ax3 = axes[1, 0]
        thresholds = np.arange(0.01, 1.0, 0.01)
        costs = []
        for thresh in thresholds:
            y_pred = (self.y_prob >= thresh).astype(int)
            cm = confusion_matrix(self.y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                total_cost = 10 * fp + 500 * fn
                costs.append(total_cost)
            else:
                costs.append(np.nan)

        ax3.plot(thresholds, costs, 'b-', linewidth=2)
        if 'cost' in self.optimal_thresholds_:
            opt_thresh = self.optimal_thresholds_['cost']['threshold']
            opt_cost = self.optimal_thresholds_['cost']['cost']
            ax3.axvline(x=opt_thresh, color='red', linestyle='--',
                       label=f'Optimal ({opt_thresh:.3f}, ${opt_cost:,.0f})')
            ax3.scatter([opt_thresh], [opt_cost], color='red', s=100, zorder=5)

        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('Total Cost ($)')
        ax3.set_title('Business Cost vs Threshold (FP=$10, FN=$500)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Threshold Comparison Summary
        ax4 = axes[1, 1]
        summary_df = self.get_all_optimal_thresholds()

        methods = summary_df['method'].tolist()
        thresholds = summary_df['threshold'].tolist()

        y_pos = np.arange(len(methods))
        colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

        ax4.barh(y_pos, thresholds, color=colors)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(methods)
        ax4.set_xlabel('Threshold')
        ax4.set_title('Optimal Thresholds by Method')
        ax4.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Default (0.5)')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='x')

        # Add threshold values as text
        for i, (thresh, method) in enumerate(zip(thresholds, methods)):
            if thresh is not None:
                ax4.text(thresh + 0.02, i, f'{thresh:.3f}', va='center')

        plt.tight_layout()
        return fig


def get_prediction_categories(y_prob: np.ndarray,
                             thresholds: dict = None) -> pd.Series:
    """
    Categorize predictions into action levels based on multiple thresholds.

    Args:
        y_prob: Predicted probabilities
        thresholds: Dictionary with threshold values

    Returns:
        Series with action categories
    """
    if thresholds is None:
        thresholds = {
            'auto_block': 0.80,
            'manual_review': 0.50,
            'monitor': 0.20
        }

    categories = pd.Series('APPROVE', index=range(len(y_prob)))
    categories[y_prob >= thresholds['monitor']] = 'MONITOR'
    categories[y_prob >= thresholds['manual_review']] = 'MANUAL_REVIEW'
    categories[y_prob >= thresholds['auto_block']] = 'AUTO_BLOCK'

    return categories


def optimize_threshold_quick(y_true, y_prob, method: str = 'f1') -> float:
    """
    Quick threshold optimization.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        method: 'f1', 'f2', 'youden', or 'cost'

    Returns:
        Optimal threshold
    """
    optimizer = ThresholdOptimizer(y_true, y_prob)

    if method == 'f1':
        threshold, _ = optimizer.optimize_f1()
    elif method == 'f2':
        threshold, _ = optimizer.optimize_fbeta(beta=2.0)
    elif method == 'youden':
        threshold, _ = optimizer.optimize_youden_j()
    elif method == 'cost':
        threshold, _ = optimizer.optimize_cost()
    else:
        threshold = 0.5

    return threshold
