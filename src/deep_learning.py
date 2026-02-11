"""
Deep Learning Models for Credit Card Fraud Detection.

Models:
1. Autoencoder - Anomaly detection via reconstruction error
2. Simple Neural Network - Classification
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not installed. Deep learning models unavailable.")
    print("Install with: pip install tensorflow")


class FraudAutoencoder(BaseEstimator, ClassifierMixin):
    """
    Autoencoder for fraud detection via anomaly detection.

    Concept:
    - Train autoencoder only on NORMAL transactions
    - Fraud transactions have higher reconstruction error
    - Threshold on reconstruction error for classification

    Architecture:
    Input -> Encoder -> Latent Space -> Decoder -> Reconstruction
    """

    def __init__(self, encoding_dim: int = 14,
                 hidden_dims: list = None,
                 threshold_percentile: float = 95,
                 epochs: int = 100,
                 batch_size: int = 256,
                 learning_rate: float = 0.001,
                 random_state: int = 42):
        """
        Initialize autoencoder.

        Args:
            encoding_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions
            threshold_percentile: Percentile of reconstruction error for threshold
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            random_state: Random seed
        """
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.threshold_percentile = threshold_percentile
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.model_ = None
        self.encoder_ = None
        self.scaler_ = None
        self.threshold_ = None
        self.history_ = None
        self.input_dim_ = None

    def _build_model(self, input_dim: int):
        """Build autoencoder model."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for Autoencoder")

        tf.random.set_seed(self.random_state)

        # Encoder
        inputs = keras.Input(shape=(input_dim,))
        x = inputs

        for dim in self.hidden_dims:
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)

        # Latent space
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='encoding')(x)

        # Decoder
        x = encoded
        for dim in reversed(self.hidden_dims):
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)

        # Output (reconstruction)
        decoded = layers.Dense(input_dim, activation='linear')(x)

        # Full autoencoder
        self.model_ = Model(inputs, decoded)
        self.encoder_ = Model(inputs, encoded)

        # Compile
        self.model_.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )

    def fit(self, X, y=None):
        """
        Fit autoencoder on NORMAL transactions only.

        Args:
            X: Features (will filter to normal class)
            y: Labels (0=normal, 1=fraud)

        Returns:
            self
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required")

        X = np.array(X)
        y = np.array(y) if y is not None else np.zeros(len(X))

        # Scale data
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Filter to normal transactions only
        X_normal = X_scaled[y == 0]

        self.input_dim_ = X_scaled.shape[1]
        self._build_model(self.input_dim_)

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        # Train
        self.history_ = self.model_.fit(
            X_normal, X_normal,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1
        )

        # Calculate threshold based on normal data reconstruction error
        reconstructions = self.model_.predict(X_normal, verbose=0)
        reconstruction_errors = np.mean(np.square(X_normal - reconstructions), axis=1)
        self.threshold_ = np.percentile(reconstruction_errors, self.threshold_percentile)

        print(f"\nAutoencoder trained on {len(X_normal)} normal transactions")
        print(f"Reconstruction error threshold (p{self.threshold_percentile}): {self.threshold_:.6f}")

        return self

    def predict_proba(self, X) -> np.ndarray:
        """
        Get fraud probability based on reconstruction error.

        Higher reconstruction error = higher fraud probability.
        """
        if self.model_ is None:
            raise ValueError("Model not fitted")

        X = np.array(X)
        X_scaled = self.scaler_.transform(X)

        # Get reconstruction
        reconstructions = self.model_.predict(X_scaled, verbose=0)

        # Calculate reconstruction error
        errors = np.mean(np.square(X_scaled - reconstructions), axis=1)

        # Convert to probability (sigmoid-like mapping)
        # Higher error = higher fraud probability
        prob_fraud = 1 / (1 + np.exp(-(errors - self.threshold_) / self.threshold_))

        return np.column_stack([1 - prob_fraud, prob_fraud])

    def predict(self, X) -> np.ndarray:
        """Predict using threshold on reconstruction error."""
        if self.model_ is None:
            raise ValueError("Model not fitted")

        X = np.array(X)
        X_scaled = self.scaler_.transform(X)

        reconstructions = self.model_.predict(X_scaled, verbose=0)
        errors = np.mean(np.square(X_scaled - reconstructions), axis=1)

        return (errors > self.threshold_).astype(int)

    def get_reconstruction_error(self, X) -> np.ndarray:
        """Get reconstruction error for each sample."""
        if self.model_ is None:
            raise ValueError("Model not fitted")

        X = np.array(X)
        X_scaled = self.scaler_.transform(X)

        reconstructions = self.model_.predict(X_scaled, verbose=0)
        return np.mean(np.square(X_scaled - reconstructions), axis=1)

    def get_encoding(self, X) -> np.ndarray:
        """Get latent space encoding."""
        if self.encoder_ is None:
            raise ValueError("Model not fitted")

        X = np.array(X)
        X_scaled = self.scaler_.transform(X)
        return self.encoder_.predict(X_scaled, verbose=0)


class FraudNeuralNetwork(BaseEstimator, ClassifierMixin):
    """
    Simple Neural Network for fraud classification.

    Architecture:
    Input -> Hidden Layers -> Output (Sigmoid)
    """

    def __init__(self, hidden_dims: list = None,
                 epochs: int = 100,
                 batch_size: int = 256,
                 learning_rate: float = 0.001,
                 class_weight: dict = None,
                 random_state: int = 42):
        """
        Initialize neural network.

        Args:
            hidden_dims: List of hidden layer dimensions
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            class_weight: Class weights for imbalanced data
            random_state: Random seed
        """
        self.hidden_dims = hidden_dims or [64, 32, 16]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.class_weight = class_weight
        self.random_state = random_state

        self.model_ = None
        self.scaler_ = None
        self.history_ = None

    def _build_model(self, input_dim: int):
        """Build neural network model."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required")

        tf.random.set_seed(self.random_state)

        model = keras.Sequential()
        model.add(keras.Input(shape=(input_dim,)))

        for i, dim in enumerate(self.hidden_dims):
            model.add(layers.Dense(dim, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(0.3))

        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )

        self.model_ = model

    def fit(self, X, y):
        """Fit neural network."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required")

        X = np.array(X)
        y = np.array(y)

        # Scale
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Build model
        self._build_model(X_scaled.shape[1])

        # Calculate class weights if not provided
        if self.class_weight is None:
            neg_count = np.sum(y == 0)
            pos_count = np.sum(y == 1)
            self.class_weight = {
                0: 1.0,
                1: neg_count / pos_count if pos_count > 0 else 1.0
            }

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_auc',
                patience=10,
                mode='max',
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        # Train
        self.history_ = self.model_.fit(
            X_scaled, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            class_weight=self.class_weight,
            callbacks=callbacks,
            verbose=1
        )

        return self

    def predict_proba(self, X) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model_ is None:
            raise ValueError("Model not fitted")

        X = np.array(X)
        X_scaled = self.scaler_.transform(X)

        prob_fraud = self.model_.predict(X_scaled, verbose=0).flatten()
        return np.column_stack([1 - prob_fraud, prob_fraud])

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """Predict with threshold."""
        prob = self.predict_proba(X)[:, 1]
        return (prob >= threshold).astype(int)


def train_autoencoder(X_train, y_train, X_test, y_test,
                     encoding_dim: int = 14, epochs: int = 100) -> Tuple[FraudAutoencoder, dict]:
    """
    Train autoencoder and evaluate.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        encoding_dim: Latent space dimension
        epochs: Training epochs

    Returns:
        Tuple of (trained_model, metrics)
    """
    if not TF_AVAILABLE:
        print("TensorFlow not available. Skipping autoencoder.")
        return None, {}

    model = FraudAutoencoder(
        encoding_dim=encoding_dim,
        epochs=epochs,
        threshold_percentile=95
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    from sklearn.metrics import roc_auc_score

    metrics = {
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }

    print("\nAutoencoder Results:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    return model, metrics


def train_neural_network(X_train, y_train, X_test, y_test,
                        hidden_dims: list = None, epochs: int = 100) -> Tuple[FraudNeuralNetwork, dict]:
    """
    Train neural network and evaluate.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        hidden_dims: Hidden layer dimensions
        epochs: Training epochs

    Returns:
        Tuple of (trained_model, metrics)
    """
    if not TF_AVAILABLE:
        print("TensorFlow not available. Skipping neural network.")
        return None, {}

    model = FraudNeuralNetwork(
        hidden_dims=hidden_dims or [64, 32, 16],
        epochs=epochs
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    from sklearn.metrics import roc_auc_score

    metrics = {
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }

    print("\nNeural Network Results:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    return model, metrics
