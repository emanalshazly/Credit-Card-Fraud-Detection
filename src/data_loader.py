"""
Data loading and preprocessing utilities for Credit Card Fraud Detection.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the credit card fraud dataset.

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame with the loaded data
    """
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} transactions, {df.shape[1]} features")
    return df


def preprocess_data(df: pd.DataFrame, scale_amount: bool = True, scale_time: bool = True) -> pd.DataFrame:
    """
    Preprocess the credit card fraud dataset.

    Args:
        df: Input DataFrame
        scale_amount: Whether to scale the Amount column
        scale_time: Whether to scale the Time column

    Returns:
        Preprocessed DataFrame
    """
    df = df.copy()

    scaler = StandardScaler()

    if scale_amount and 'Amount' in df.columns:
        df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
        df = df.drop('Amount', axis=1)

    if scale_time and 'Time' in df.columns:
        df['Time_scaled'] = scaler.fit_transform(df[['Time']])
        df = df.drop('Time', axis=1)

    return df


def split_data(df: pd.DataFrame, target_col: str = 'Class',
               test_size: float = 0.2, random_state: int = 42):
    """
    Split data into training and testing sets.

    Args:
        df: Input DataFrame
        target_col: Name of the target column
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    print(f"Fraud ratio in train: {y_train.sum() / len(y_train) * 100:.3f}%")
    print(f"Fraud ratio in test: {y_test.sum() / len(y_test) * 100:.3f}%")

    return X_train, X_test, y_train, y_test


def get_data_summary(df: pd.DataFrame, target_col: str = 'Class') -> dict:
    """
    Get summary statistics of the dataset.

    Args:
        df: Input DataFrame
        target_col: Name of the target column

    Returns:
        Dictionary with summary statistics
    """
    normal = df[df[target_col] == 0]
    fraud = df[df[target_col] == 1]

    summary = {
        'total_transactions': len(df),
        'normal_transactions': len(normal),
        'fraud_transactions': len(fraud),
        'fraud_percentage': len(fraud) / len(df) * 100,
        'normal_percentage': len(normal) / len(df) * 100,
        'imbalance_ratio': len(normal) / len(fraud) if len(fraud) > 0 else float('inf'),
        'features': list(df.columns),
        'missing_values': df.isnull().sum().sum()
    }

    return summary
