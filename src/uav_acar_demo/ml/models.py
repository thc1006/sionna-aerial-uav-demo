from __future__ import annotations

"""Simple Keras models for Phase 1 ML experiments.

The idea is to start with very lightweight baselines that map Sionna/pyAerial
features (e.g., RSRP/SNR grids, interference indicators) to throughput or
outage probability. These are not meant to be production architectures; they
are just easy starting points.
"""

from typing import Sequence, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models


def build_dense_baseline(input_dim: int, num_outputs: int = 1) -> tf.keras.Model:
    """Return a small fully connected network for tabular features."""
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_outputs),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def build_cnn_baseline(
    input_shape: Tuple[int, int, int], num_outputs: int = 1
) -> tf.keras.Model:
    """Return a small CNN for 2D feature maps (e.g., RB x time grids)."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_outputs)(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    return model
