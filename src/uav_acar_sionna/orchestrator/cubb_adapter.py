from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple


def run_emulation_with_cubb(
    h5_for_cubb: Path,
    h5_for_ru: Path,
    snr_db: float,
    sinr_db: float,
    log_path: Path,
) -> Tuple[float, float]:
    """Stub cuBB adapter that 'runs' an emulation and writes a log.

    In the real system this should:
    - Launch testMAC + cuPHY + RU emulator containers with the provided TVs.
    - Wait for completion, then parse official throughput / BLER logs.
    - Return the measured throughput and BLER.

    For now we simulate a simple throughput curve based on SINR using a
    Shannon-like formula and write a tiny text log.

    Parameters
    ----------
    h5_for_cubb, h5_for_ru:
        Paths to the TVs we would feed into testMAC and RU emulator.
    snr_db, sinr_db:
        Link quality estimates from SionnaRT.
    log_path:
        Where to write the synthetic throughput log.

    Returns
    -------
    throughput_mbps, bler : float
        Synthetic KPIs that mimic the shape of a realistic curve.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Very rough synthetic throughput: convert SINR (dB) to linear, apply log2.
    sinr_lin = 10 ** (sinr_db / 10.0)
    spectral_eff = math.log2(1.0 + sinr_lin)  # bits/s/Hz

    # Assume 20 MHz bandwidth for now (can be replaced later).
    bandwidth_hz = 20e6
    throughput_bps = spectral_eff * bandwidth_hz
    throughput_mbps = throughput_bps / 1e6

    # Make up a BLER that decreases with SINR.
    bler = max(0.0, min(1.0, 0.3 * math.exp(-0.15 * (sinr_db - 5.0))))

    timestamp = datetime.now(timezone.utc).isoformat()

    log_path.write_text(
        f"""timestamp={timestamp}
h5_for_cubb={h5_for_cubb}
h5_for_ru={h5_for_ru}
snr_db={snr_db:.2f}
sinr_db={sinr_db:.2f}
throughput_mbps={throughput_mbps:.3f}
bler={bler:.5f}
""",
        encoding="utf-8",
    )

    return throughput_mbps, bler
