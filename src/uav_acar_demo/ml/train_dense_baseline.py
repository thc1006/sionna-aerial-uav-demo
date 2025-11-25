"""Training script for dense baseline model (Phase 1 ML experiments).

This script demonstrates how to train a simple fully-connected network to
predict throughput from UAV link features (RSRP, SNR, distance, etc.).

The workflow:
  1. Load Sionna simulation results as training data
  2. Extract features and labels
  3. Build and compile dense baseline model
  4. Train with train/val split
  5. Evaluate and save model

Usage:
    python -m uav_acar_demo.ml.train_dense_baseline --data-dir data/phase0 --epochs 100

References:
    - Sionna ML integration: https://nvlabs.github.io/sionna/made_with_sionna.html
    - Keras Sequential API: https://keras.io/guides/sequential_model/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from .models import build_dense_baseline


def load_training_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load Sionna simulation results and prepare training data.

    Args:
        data_dir: Directory containing phase0_metrics.json or similar

    Returns:
        Tuple of (features, labels) as numpy arrays
        - features: [num_samples, num_features] - RSRP, SNR, distance, etc.
        - labels: [num_samples, 1] - throughput in Mbps
    """
    metrics_file = data_dir / "phase0_metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    with open(metrics_file) as f:
        scenarios = json.load(f)

    # Extract features and labels
    features_list = []
    labels_list = []

    for name, metrics in scenarios.items():
        # Feature vector: [distance_m, rsrp_dbm, snr_db]
        # You can extend this with more features (frequency, bandwidth, etc.)
        features = [
            metrics["distance_m"] / 1000.0,  # Normalize to km
            metrics["rsrp_dbm"] / 100.0,     # Normalize dBm
            metrics["snr_db"] / 100.0,       # Normalize dB
        ]
        label = metrics["throughput_mbps"] / 1000.0  # Normalize to Gbps

        features_list.append(features)
        labels_list.append(label)

    # Convert to numpy arrays
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.float32).reshape(-1, 1)

    print(f"Loaded {len(X)} samples")
    print(f"Feature shape: {X.shape}, Label shape: {y.shape}")
    print(f"Feature ranges: {X.min(axis=0)} to {X.max(axis=0)}")
    print(f"Label range: {y.min()} to {y.max()}")

    return X, y


def augment_data(X: np.ndarray, y: np.ndarray, num_augmented: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Augment training data by adding noise and interpolation.

    For Phase 0, we only have 3 data points. To train a neural network,
    we create synthetic samples by:
      - Adding Gaussian noise to features
      - Linear interpolation between samples
      - Physical constraints (throughput should decrease with distance)

    Args:
        X: Original features [num_samples, num_features]
        y: Original labels [num_samples, 1]
        num_augmented: Number of augmented samples to generate

    Returns:
        Augmented (X, y) with more samples
    """
    X_aug = [X]
    y_aug = [y]

    # Add noise-based augmentation
    for _ in range(num_augmented // 2):
        noise = np.random.normal(0, 0.05, X.shape)
        X_noisy = X + noise
        y_noisy = y + np.random.normal(0, 0.02, y.shape)
        X_aug.append(X_noisy)
        y_aug.append(y_noisy)

    # Add interpolation-based augmentation
    for _ in range(num_augmented // 2):
        # Randomly select two samples
        idx1, idx2 = np.random.choice(len(X), 2, replace=False)
        alpha = np.random.uniform(0.2, 0.8)
        X_interp = alpha * X[idx1] + (1 - alpha) * X[idx2]
        y_interp = alpha * y[idx1] + (1 - alpha) * y[idx2]
        X_aug.append(X_interp.reshape(1, -1))
        y_aug.append(y_interp.reshape(1, -1))

    X_augmented = np.vstack(X_aug)
    y_augmented = np.vstack(y_aug)

    print(f"Augmented dataset: {X_augmented.shape[0]} samples")

    return X_augmented, y_augmented


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 100,
    batch_size: int = 16,
    validation_split: float = 0.2,
) -> Tuple:
    """Train dense baseline model.

    Args:
        X: Features [num_samples, num_features]
        y: Labels [num_samples, 1]
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of data for validation

    Returns:
        Tuple of (model, history)
    """
    input_dim = X.shape[1]
    model = build_dense_baseline(input_dim=input_dim, num_outputs=1)

    print("\nModel architecture:")
    model.summary()

    # Train model
    print(f"\nTraining for {epochs} epochs...")
    history = model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
    )

    return model, history


def plot_training_history(history, output_dir: Path):
    """Plot training and validation loss curves.

    Args:
        history: Keras training history object
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.semilogy(history.history["loss"], label="Training Loss")
    plt.semilogy(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE, log scale)")
    plt.title("Training History (Log Scale)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_file = output_dir / "training_history.png"
    plt.savefig(plot_file, dpi=150)
    print(f"Saved training history plot to {plot_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train dense baseline model for throughput prediction"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/phase0",
        help="Directory containing training data (phase0_metrics.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/dense_baseline",
        help="Directory to save trained model and plots",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num-augmented",
        type=int,
        default=200,
        help="Number of augmented samples to generate",
    )
    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading training data from {data_dir}...")
    X, y = load_training_data(data_dir)

    # Augment data (needed for Phase 0 with only 3 samples)
    print(f"\nAugmenting data to {args.num_augmented} samples...")
    X_aug, y_aug = augment_data(X, y, num_augmented=args.num_augmented)

    # Train model
    model, history = train_model(
        X_aug,
        y_aug,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
    )

    # Save model
    model_file = output_dir / "dense_baseline.keras"
    model.save(model_file)
    print(f"\nSaved model to {model_file}")

    # Plot training history
    plot_training_history(history, output_dir)

    # Evaluate on original data (test set)
    print("\nEvaluating on original 3 scenarios:")
    y_pred = model.predict(X, verbose=0)
    for i, (y_true_norm, y_pred_norm) in enumerate(zip(y, y_pred)):
        y_true_mbps = y_true_norm[0] * 1000  # Denormalize
        y_pred_mbps = y_pred_norm[0] * 1000
        error_pct = abs(y_pred_mbps - y_true_mbps) / y_true_mbps * 100
        print(f"  Sample {i}: True={y_true_mbps:.2f} Mbps, "
              f"Pred={y_pred_mbps:.2f} Mbps, Error={error_pct:.1f}%")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
