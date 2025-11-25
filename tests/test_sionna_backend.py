
from uav_acar_sionna.config import UavScenarioConfig
from uav_acar_sionna.sionna_backend import (
    estimate_link_with_sionna_backend,
    scenario_to_link_config,
)


def test_backend_basic_monotonicity():
    """Check that near vs far UAV scenarios follow the right trend."""
    near = UavScenarioConfig(distance_m=100.0)
    far = UavScenarioConfig(distance_m=1000.0)

    cfg_near = scenario_to_link_config(near)
    cfg_far = scenario_to_link_config(far)

    # Closer UAV should see higher SNR
    assert cfg_near.snr_db > cfg_far.snr_db

    res_near = estimate_link_with_sionna_backend(near)
    res_far = estimate_link_with_sionna_backend(far)

    # And better throughput as well
    assert res_near.throughput_mbps >= res_far.throughput_mbps
