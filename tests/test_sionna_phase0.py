from __future__ import annotations

from uav_acar_demo.sim.generate_link_metrics import (
    simulate_link_metrics,
    default_scenarios,
)


def test_far_has_lower_metrics_than_near() -> None:
    scenarios = default_scenarios()
    short = simulate_link_metrics(scenarios.short)
    mid = simulate_link_metrics(scenarios.mid)
    far = simulate_link_metrics(scenarios.far)

    # RSRP and SNR should decay monotonically with distance
    assert short["distance_m"] < mid["distance_m"] < far["distance_m"]
    assert short["rsrp_dbm"] > mid["rsrp_dbm"] > far["rsrp_dbm"]
    assert short["snr_db"] > mid["snr_db"] > far["snr_db"]
    assert short["throughput_mbps"] > mid["throughput_mbps"] > far["throughput_mbps"]
