"""
Minimal SISO Rayleigh block-fading link estimator using TensorFlow.

This module is intentionally self-contained and designed to be easy to
adapt into a larger project (e.g. an ACAR/Sionna bridge).  It assumes
that TensorFlow is installed and (optionally) that a GPU such as an
RTX 4090 is available, but it does not depend on any other parts of
Sionna.  The Rayleigh fading itself is implemented directly using
complex Gaussian random variables.

The main entry point is :func:`estimate_link_with_sionna`, which
simulates uncoded BPSK over a flat Rayleigh channel with perfect
receiver-side CSI and returns a small dictionary of metrics.

You can use this function as a drop-in backend for an ACAR TestVector
by mapping your scenario description to (num_bits, snr_db, batch_size).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import tensorflow as tf


@dataclass
class LinkSimConfig:
    """Configuration for a single SISO Rayleigh link simulation.

    Parameters
    ----------
    num_bits :
        Number of information bits per batch element.
    snr_db :
        Signal-to-noise ratio in dB at the receiver (per symbol).
    batch_size :
        Number of independent realizations to simulate in parallel.
    seed :
        Random seed for TensorFlow's RNG for reproducibility.
    """

    num_bits: int = 4096
    snr_db: float = 10.0
    batch_size: int = 1
    seed: int = 0


def _bpsk_modulate(bits: tf.Tensor) -> tf.Tensor:
    """Map {0,1} bits to BPSK symbols {+1,-1} (real-valued).

    bits: int32/64 tensor of shape [..., num_bits]
    returns: complex64 tensor of same shape (imag part = 0)
    """
    bits = tf.cast(bits, tf.float32)
    # 0 -> +1, 1 -> -1
    symbols = 1.0 - 2.0 * bits
    return tf.cast(symbols, tf.complex64)


def _add_rayleigh_awgn(
    x: tf.Tensor,
    snr_db: float,
    rng: tf.random.Generator,
):
    """Apply SISO Rayleigh block fading with AWGN.

    Parameters
    ----------
    x :
        Transmitted symbols, complex64 tensor of shape [B, N].
    snr_db :
        SNR per symbol in dB.
    rng :
        tf.random.Generator used for reproducible randomness.

    Returns
    -------
    y :
        Received symbols after Rayleigh fading and AWGN.
    h :
        Channel coefficients (Rayleigh), shape [B, N].
    """
    x = tf.cast(x, tf.complex64)
    snr_linear = tf.pow(10.0, snr_db / 10.0)

    # Rayleigh fading coefficient h ~ CN(0,1)
    real = rng.normal(shape=tf.shape(x), dtype=tf.float32)
    imag = rng.normal(shape=tf.shape(x), dtype=tf.float32)
    h = tf.complex(real, imag) / tf.cast(tf.sqrt(2.0), tf.complex64)

    # Scale noise to achieve desired SNR per symbol.
    # For unit-power BPSK symbols and unit-variance channel, the
    # average received signal power is E[|h x|^2] = 1.  Hence
    # noise variance per complex dimension is sigma^2 = 1 / (2 * SNR).
    noise_sigma = tf.sqrt(1.0 / (2.0 * snr_linear))
    n_real = rng.normal(shape=tf.shape(x), dtype=tf.float32)
    n_imag = rng.normal(shape=tf.shape(x), dtype=tf.float32)
    n = tf.complex(n_real, n_imag) * tf.cast(noise_sigma, tf.complex64)

    y = h * x + n
    return y, h


def _bpsk_hard_demod(x_hat: tf.Tensor) -> tf.Tensor:
    """Hard-decision BPSK demapper.

    x_hat: complex64 tensor of shape [B, N]
    returns: int32 tensor of bits in {0,1} with same shape.
    """
    # Decision based on real part sign
    decisions = tf.math.real(x_hat)
    bits_hat = tf.cast(decisions < 0.0, tf.int32)
    return bits_hat


def estimate_link_with_sionna(config: LinkSimConfig) -> Dict[str, float]:
    """Run a minimal SISO Rayleigh + AWGN link simulation.

    The simulation follows these steps:

    1. Generate i.i.d. Bernoulli(0.5) bits.
    2. BPSK-modulate the bits to symbols x in {+1,-1}.
    3. Apply SISO Rayleigh block fading + AWGN to obtain y.
    4. Perform perfect-CSI equalization: x̂ = y / h.
    5. Demap back to bits and compute the BER.
    6. Report a simple "throughput" proxy = (1 - BER) [bits/use].

    This is deliberately simple but uses the same mathematical
    Rayleigh model that Sionna's RayleighBlockFading relies on,
    so it is a good sanity check for your end-to-end plumbing.

    Parameters
    ----------
    config :
        :class:`LinkSimConfig` instance specifying SNR, num_bits, etc.

    Returns
    -------
    metrics :
        Dict with the following keys:

        * ``ber`` – Bit error rate (float).
        * ``throughput_bits_per_use`` – Simple proxy for throughput,
          numerically equal to ``1 - ber`` for BPSK over a SISO link.
        * ``snr_db`` – Echo of the input SNR in dB.
        * ``num_bits`` – Number of bits simulated (per batch element).
    """
    tf.random.set_seed(config.seed)
    rng = tf.random.Generator.from_seed(config.seed)

    # 1) Generate random bits [B, N]
    bits = rng.uniform(
        shape=[config.batch_size, config.num_bits],
        minval=0,
        maxval=2,
        dtype=tf.int32,
    )

    # 2) BPSK modulation
    x = _bpsk_modulate(bits)  # [B, N]

    # 3) Rayleigh fading + AWGN
    y, h = _add_rayleigh_awgn(x, config.snr_db, rng)

    # 4) Perfect CSI equalization.
    #    Avoid division by zero by adding a tiny epsilon to |h|^2.
    eps = tf.cast(1e-9, tf.float32)
    h_conj = tf.math.conj(h)
    denom = tf.cast(tf.math.abs(h) ** 2, tf.float32) + eps
    denom = tf.cast(denom, tf.complex64)
    x_hat = y * h_conj / denom

    # 5) Hard-decision demapping and BER
    bits_hat = _bpsk_hard_demod(x_hat)
    # Flatten over batch for the BER calculation
    bits_flat = tf.reshape(bits, [-1])
    bits_hat_flat = tf.reshape(bits_hat, [-1])
    bit_errors = tf.reduce_sum(
        tf.cast(bits_flat != bits_hat_flat, tf.float32)
    )
    num_bits_total = tf.cast(tf.size(bits_flat), tf.float32)
    ber = bit_errors / num_bits_total

    # 6) Throughput proxy: good bits / channel use
    throughput_bits_per_use = 1.0 - ber

    # Convert to Python floats for a lightweight JSON / DB interface
    return {
        "ber": float(ber.numpy()),
        "throughput_bits_per_use": float(throughput_bits_per_use.numpy()),
        "snr_db": float(config.snr_db),
        "num_bits": int(config.num_bits),
    }
