"""Tests for minimal Sionna link estimator and UAV bridge."""

import pytest

from uav_acar_demo.config import UavScenarioConfig
from uav_acar_demo.sim import (
    LinkSimConfig,
    estimate_link_with_sionna,
    run_minimal_link_sim,
    compare_uma_vs_rayleigh,
    default_scenarios,
)


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


def test_uav_bridge_integration():
    """Test that UAV scenario config can be used with minimal link estimator."""
    scenarios = default_scenarios()

    # Test with short distance scenario
    result = run_minimal_link_sim(scenarios.short, seed=42)

    # Verify all expected keys are present
    assert "ber" in result
    assert "throughput_bits_per_use" in result
    assert "snr_db" in result
    assert "num_bits" in result
    assert "distance_m" in result
    assert "scenario_name" in result

    # Verify values are reasonable
    assert 0.0 <= result["ber"] <= 1.0
    assert 0.0 <= result["throughput_bits_per_use"] <= 1.0
    assert result["distance_m"] == scenarios.short.distance_m
    assert result["scenario_name"] == "short"


def test_compare_uma_vs_rayleigh():
    """Test comparison function between UMa and Rayleigh models."""
    cfg = UavScenarioConfig(
        name="test",
        tx_height_m=25.0,
        ue_height_m=120.0,
        distance_m=500.0,
    )

    comparison = compare_uma_vs_rayleigh(cfg, seed=42)

    # Verify structure
    assert "uma" in comparison
    assert "rayleigh" in comparison

    # Verify UMa results
    uma = comparison["uma"]
    assert "rsrp_dbm" in uma
    assert "snr_db" in uma
    assert "throughput_mbps" in uma
    assert "distance_m" in uma

    # Verify Rayleigh results
    rayleigh = comparison["rayleigh"]
    assert "ber" in rayleigh
    assert "throughput_bits_per_use" in rayleigh
    assert "snr_db" in rayleigh

    # Both should use same SNR (from UMa)
    assert pytest.approx(uma["snr_db"], rel=0.01) == pytest.approx(rayleigh["snr_db"], rel=0.01)

    # Rayleigh BER should be reasonable at this SNR
    assert 0.0 <= rayleigh["ber"] <= 1.0


def test_three_distance_scenarios_with_link_estimator():
    """Test that all three UAV distance scenarios work with link estimator."""
    scenarios = default_scenarios()

    results = {}
    for scenario_cfg in [scenarios.short, scenarios.mid, scenarios.far]:
        result = run_minimal_link_sim(scenario_cfg, seed=42)
        results[scenario_cfg.name] = result

    # Verify all scenarios completed
    assert len(results) == 3
    assert "short" in results
    assert "mid" in results
    assert "far" in results

    # All should have valid BER
    for name, result in results.items():
        assert 0.0 <= result["ber"] <= 1.0, f"{name}: BER out of range"
        assert 0.0 <= result["throughput_bits_per_use"] <= 1.0, f"{name}: Throughput out of range"
