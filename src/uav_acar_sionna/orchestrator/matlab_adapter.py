from __future__ import annotations

from pathlib import Path

import h5py


def generate_testvector_h5(npz_path: Path, h5_path: Path) -> None:
    """Stub MATLAB adapter that creates a minimal `.h5` file for cuBB.

    In the real pipeline, this function should:
    - Call MATLAB (or the Aerial Python mcore module) to transform the SionnaRT
      channel data stored in `npz_path` into a cuBB-compatible TestVector `.h5`.
    - Populate all the datasets and attributes expected by testMAC/cuPHY/RU.

    For now, we only create a tiny HDF5 file with a few attributes so that
    downstream scripts that expect a `.h5` can at least open the file.
    """
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["source_npz"] = str(npz_path)
        meta.attrs["note"] = (
            "Dummy TestVector file. Replace generate_testvector_h5() with "
            "a real MATLAB / Aerial TV generation pipeline."
        )
