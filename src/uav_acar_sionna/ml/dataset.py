"""
PyTorch datasets for UAV channel measurement data.
"""

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class SummaryDataset(Dataset):
    """
    PyTorch Dataset that reads summary.csv for tabular features.

    Loads features (distance_m, snr_db, sinr_db) and targets (throughput_mbps, bler).
    Optionally loads channel data (h_real, h_imag) from NPZ files if channel_data=True.

    Parameters
    ----------
    csv_path : Path or str
        Path to summary.csv file
    target : {"throughput_mbps", "bler"}
        Which column to use as prediction target
    channel_data : bool, optional
        If True, load channel matrices (h_real, h_imag) from NPZ files

    Attributes
    ----------
    df : pd.DataFrame
        Full dataframe from summary.csv
    features : torch.Tensor
        Feature tensor of shape (N, 3) with [distance_m, snr_db, sinr_db]
    targets : torch.Tensor
        Target tensor of shape (N,) with throughput_mbps or bler values
    channel_data : bool
        Whether channel data is loaded
    npz_paths : list of Path or None
        Paths to NPZ files if channel_data=True
    """

    def __init__(
        self,
        csv_path: Path | str,
        target: Literal["throughput_mbps", "bler"] = "throughput_mbps",
        channel_data: bool = False,
    ):
        self.csv_path = Path(csv_path)
        self.target = target
        self.channel_data = channel_data

        # Load CSV
        self.df = pd.read_csv(self.csv_path)

        # Extract tabular features: distance_m, snr_db, sinr_db
        feature_cols = ["distance_m", "snr_db", "sinr_db"]
        features_np = self.df[feature_cols].values.astype(np.float32)
        self.features = torch.from_numpy(features_np)

        # Extract targets
        targets_np = self.df[target].values.astype(np.float32)
        self.targets = torch.from_numpy(targets_np)

        # Store NPZ paths if we need channel data
        if self.channel_data:
            self.npz_paths = [Path(p) for p in self.df["sionna_npz_path"]]
        else:
            self.npz_paths = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns
        -------
        dict with keys:
            - "features": torch.Tensor of shape (3,) with [distance_m, snr_db, sinr_db]
            - "target": torch.Tensor scalar
            - "h_complex": torch.Tensor of shape (4, 4096) (only if channel_data=True)
        """
        item = {
            "features": self.features[idx],
            "target": self.targets[idx],
        }

        if self.channel_data and self.npz_paths is not None:
            # Load channel data from NPZ if available
            npz_path = self.npz_paths[idx]
            data = np.load(npz_path)

            if "h_real" in data and "h_imag" in data:
                h_real = data["h_real"]  # shape: (4, 4096)
                h_imag = data["h_imag"]  # shape: (4, 4096)

                # Combine into complex tensor
                h_complex = torch.from_numpy(h_real) + 1j * torch.from_numpy(h_imag)
                item["h_complex"] = h_complex
            else:
                # If channel data is missing, create a zero tensor
                item["h_complex"] = torch.zeros(4, 4096, dtype=torch.complex64)

        return item


def create_dataloader(
    csv_path: Path | str,
    target: Literal["throughput_mbps", "bler"] = "throughput_mbps",
    channel_data: bool = False,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a PyTorch DataLoader from summary.csv.

    Parameters
    ----------
    csv_path : Path or str
        Path to summary.csv file
    target : {"throughput_mbps", "bler"}
        Which column to use as prediction target
    channel_data : bool
        If True, load channel matrices from NPZ files
    batch_size : int
        Batch size for DataLoader
    shuffle : bool
        Whether to shuffle data
    num_workers : int
        Number of worker processes for data loading

    Returns
    -------
    DataLoader
        PyTorch DataLoader ready for training/evaluation
    """
    dataset = SummaryDataset(
        csv_path=csv_path,
        target=target,
        channel_data=channel_data,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
