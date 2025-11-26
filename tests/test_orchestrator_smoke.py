from __future__ import annotations

from uav_acar_sionna.orchestrator.run_emulation import EmulationConfig, run_emulation_once


def test_run_emulation_once_smoke(tmp_path, monkeypatch):
    # Run a single synthetic emulation and assert that it returns sane values.
    cfg = EmulationConfig(scenario_id="test_scenario", time_point=1.0, distance_m=300.0)
    result = run_emulation_once(cfg)

    assert result.throughput_mbps > 0.0
    assert result.snr_db > result.sinr_db
