"""
LSTM Autoencoder for time-series anomaly detection.

Architecture:
    Encoder: LSTM layers compress input sequences to a latent representation.
    Decoder: LSTM layers reconstruct the original input from the latent space.
    Anomaly score: Reconstruction error (MSE) — high error indicates anomaly.

Reference:
    Malhotra, P. et al. (2016). "LSTM-based Encoder-Decoder for Multi-Sensor
    Anomaly Detection." ICML Workshop on Anomaly Detection.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_sequences(data, sequence_length=30):
    """
    Create sliding window sequences for LSTM input.

    Converts tabular data into 3D sequences of shape:
        (n_samples, sequence_length, n_features)

    Args:
        data: 2D numpy array of shape (n_samples, n_features)
        sequence_length: number of timesteps per sequence

    Returns:
        3D numpy array of sequences
    """
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)


def build_lstm_autoencoder(n_features, sequence_length=30, latent_dim=32,
                            dropout_rate=0.2):
    """
    Build an LSTM Autoencoder for multivariate time-series anomaly detection.

    Architecture:
        Encoder:
            LSTM(64) -> Dropout -> LSTM(latent_dim) -> RepeatVector
        Decoder:
            LSTM(latent_dim) -> Dropout -> LSTM(64) -> TimeDistributed(Dense)

    The bottleneck (latent_dim) forces the model to learn a compressed
    representation of normal patterns. Anomalous inputs produce higher
    reconstruction error.

    Args:
        n_features: number of input features per timestep
        sequence_length: number of timesteps in each input sequence
        latent_dim: dimensionality of the latent space
        dropout_rate: dropout probability for regularization

    Returns:
        Compiled Keras model
    """
    # Encoder
    inputs = keras.Input(shape=(sequence_length, n_features))
    x = layers.LSTM(64, return_sequences=True, name='encoder_lstm1')(inputs)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.LSTM(latent_dim, return_sequences=False, name='encoder_lstm2')(x)
    x = layers.Dropout(dropout_rate)(x)

    # Bridge
    x = layers.RepeatVector(sequence_length, name='bridge')(x)

    # Decoder
    x = layers.LSTM(latent_dim, return_sequences=True, name='decoder_lstm1')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.LSTM(64, return_sequences=True, name='decoder_lstm2')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.TimeDistributed(layers.Dense(n_features), name='output')(x)

    model = keras.Model(inputs, outputs, name='lstm_autoencoder')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss='mse')
    return model


def build_deep_autoencoder(n_features, encoding_dims=[64, 32, 16],
                            dropout_rate=0.2):
    """
    Build a Deep (Dense) Autoencoder as a baseline comparison.

    Architecture:
        Encoder: Dense(64) -> Dense(32) -> Dense(16)
        Decoder: Dense(32) -> Dense(64) -> Dense(n_features)

    Args:
        n_features: number of input features
        encoding_dims: list of hidden layer dimensions for encoder
        dropout_rate: dropout probability

    Returns:
        Compiled Keras model
    """
    # Encoder
    inputs = keras.Input(shape=(n_features,))
    x = inputs
    for i, dim in enumerate(encoding_dims):
        x = layers.Dense(dim, activation='relu', name=f'encoder_{i}')(x)
        x = layers.Dropout(dropout_rate)(x)

    # Decoder
    for i, dim in enumerate(reversed(encoding_dims[:-1])):
        x = layers.Dense(dim, activation='relu', name=f'decoder_{i}')(x)
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(n_features, activation='linear', name='output')(x)

    model = keras.Model(inputs, outputs, name='deep_autoencoder')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss='mse')
    return model


def train_autoencoder(model, X_train, X_val=None, epochs=50, batch_size=64,
                       patience=10):
    """
    Train autoencoder with early stopping on validation loss.

    Args:
        model: compiled Keras model
        X_train: training data (normal samples only)
        X_val: validation data (normal samples only)
        epochs: maximum training epochs
        batch_size: mini-batch size
        patience: early stopping patience

    Returns:
        Trained model and training history
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        ),
    ]

    validation_data = (X_val, X_val) if X_val is not None else None

    history = model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history


def compute_reconstruction_error(model, X, per_sample=True):
    """
    Compute reconstruction error (MSE) for anomaly scoring.

    Higher reconstruction error indicates the model could not faithfully
    reproduce the input — suggesting the input deviates from the normal
    patterns learned during training.

    Args:
        model: trained autoencoder
        X: input data
        per_sample: if True, return MSE per sample; else return overall MSE

    Returns:
        Array of reconstruction errors (one per sample)
    """
    X_pred = model.predict(X, verbose=0)

    if X_pred.ndim == 3:
        # LSTM autoencoder: average over timesteps and features
        errors = np.mean(np.square(X - X_pred), axis=(1, 2))
    else:
        # Dense autoencoder: average over features
        errors = np.mean(np.square(X - X_pred), axis=1)

    return errors


def find_optimal_threshold(errors_normal, errors_all, y_true, method='f1'):
    """
    Find optimal anomaly threshold by maximizing F1-score.

    Tests thresholds across the range of reconstruction errors and
    selects the one that maximizes the F1-score on the validation set.

    Args:
        errors_normal: reconstruction errors for normal training data
        errors_all: reconstruction errors for the full evaluation set
        y_true: true labels (0=normal, 1=anomaly)
        method: optimization criterion ('f1' or 'percentile')

    Returns:
        Optimal threshold value
    """
    from sklearn.metrics import f1_score

    if method == 'percentile':
        # Use 95th percentile of normal errors as threshold
        threshold = np.percentile(errors_normal, 95)
    elif method == 'f1':
        # Grid search over thresholds
        thresholds = np.linspace(
            np.percentile(errors_all, 80),
            np.percentile(errors_all, 99.5),
            200,
        )
        best_f1, best_thresh = 0, thresholds[0]
        for t in thresholds:
            y_pred = (errors_all > t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        threshold = best_thresh
        print(f"Optimal threshold: {threshold:.6f} (F1={best_f1:.4f})")

    return threshold
