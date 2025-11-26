"""Tests for ML module."""

import pytest
import torch

from uav_acar_sionna.ml import SummaryDataset, create_dataloader, FCRegressor, CNN1D


@pytest.fixture
def csv_path():
    return "data/phase2_interference/summary.csv"


def test_dataset_loads(csv_path):
    """Test that dataset loads successfully."""
    dataset = SummaryDataset(csv_path, target="throughput_mbps")
    assert len(dataset) > 0

    sample = dataset[0]
    assert "features" in sample
    assert "target" in sample
    assert sample["features"].shape == (3,)


def test_dataloader_creates(csv_path):
    """Test that dataloader creates successfully."""
    loader = create_dataloader(csv_path, target="throughput_mbps", batch_size=4)
    batch = next(iter(loader))

    assert "features" in batch
    assert "target" in batch


def test_fc_regressor_forward():
    """Test FCRegressor forward pass."""
    model = FCRegressor(input_dim=3, hidden_dims=(64, 32))
    x = torch.randn(4, 3)
    out = model(x)

    assert out.shape == (4, 1)


def test_cnn1d_forward():
    """Test CNN1D forward pass."""
    model = CNN1D(num_antennas=4, seq_len=4096)
    # Input: (batch, num_antennas, seq_len) as complex tensor
    x = torch.randn(2, 4, 4096, dtype=torch.cfloat)
    out = model(x)

    assert out.shape == (2, 1)


def test_models_on_gpu_if_available():
    """Test that models can be moved to GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model = FCRegressor(input_dim=3)
    model = model.cuda()
    x = torch.randn(4, 3).cuda()
    out = model(x)

    assert out.device.type == "cuda"
