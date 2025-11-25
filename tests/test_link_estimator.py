"""Tests for minimal Sionna link estimator."""

import pytest

from sionna_link_minimal import LinkSimConfig, estimate_link_with_sionna


@pytest.mark.parametrize(
    "snr_db, max_ber",
    [
        (0.0, 0.3),   # Very noisy, BER can be high but should be < 0.3 for 4k bits
        (10.0, 0.1),  # Medium SNR
        (20.0, 0.02), # High SNR, expect low BER
    ],
)
def test_ber_monotonic_with_snr(snr_db, max_ber):
    """Smoke test to ensure BER decreases as SNR increases."""
    cfg = LinkSimConfig(num_bits=4096, snr_db=snr_db, batch_size=4, seed=42)
    metrics = estimate_link_with_sionna(cfg)

    assert 0.0 <= metrics["ber"] <= 1.0
    assert metrics["ber"] <= max_ber
    assert pytest.approx(metrics["snr_db"], rel=1e-6) == snr_db
    assert metrics["num_bits"] == cfg.num_bits


def test_throughput_sanity():
    """Throughput proxy 1-BER should be between 0 and 1 and improve with SNR."""
    cfg_low = LinkSimConfig(num_bits=4096, snr_db=0.0, batch_size=4, seed=123)
    cfg_high = LinkSimConfig(num_bits=4096, snr_db=20.0, batch_size=4, seed=123)

    metrics_low = estimate_link_with_sionna(cfg_low)
    metrics_high = estimate_link_with_sionna(cfg_high)

    for m in (metrics_low, metrics_high):
        assert 0.0 <= m["throughput_bits_per_use"] <= 1.0

    assert metrics_high["throughput_bits_per_use"] >= metrics_low["throughput_bits_per_use"]
