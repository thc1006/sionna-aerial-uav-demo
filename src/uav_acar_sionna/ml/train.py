"""
Training and evaluation utilities for ML models.
"""

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from uav_acar_sionna.ml.dataset import create_dataloader
from uav_acar_sionna.ml.models import FCRegressor, CNN1D


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_channel_data: bool = False,
) -> float:
    """
    Train model for one epoch.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to train
    dataloader : DataLoader
        Training data loader
    optimizer : torch.optim.Optimizer
        Optimizer
    criterion : nn.Module
        Loss function
    device : torch.device
        Device to train on (cpu or cuda)
    use_channel_data : bool
        If True, use h_complex from batch; otherwise use features

    Returns
    -------
    float
        Average loss over the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        target = batch["target"].to(device).unsqueeze(-1)  # (batch, 1)

        if use_channel_data:
            x = batch["h_complex"].to(device)  # (batch, 4, 4096)
        else:
            x = batch["features"].to(device)  # (batch, 3)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_channel_data: bool = False,
) -> dict[str, float]:
    """
    Evaluate model on validation/test set.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to evaluate
    dataloader : DataLoader
        Evaluation data loader
    criterion : nn.Module
        Loss function
    device : torch.device
        Device to evaluate on
    use_channel_data : bool
        If True, use h_complex from batch; otherwise use features

    Returns
    -------
    dict
        Dictionary with keys: loss, mae, rmse, r2
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            target = batch["target"].to(device).unsqueeze(-1)

            if use_channel_data:
                x = batch["h_complex"].to(device)
            else:
                x = batch["features"].to(device)

            pred = model(x)
            loss = criterion(pred, target)

            total_loss += loss.item()
            num_batches += 1

            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    # Compute metrics
    import numpy as np

    preds = np.concatenate(all_preds, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()

    mae = mean_absolute_error(targets, preds)
    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, preds)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    return {
        "loss": avg_loss,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def train_model(
    model_type: Literal["fc", "cnn"],
    csv_path: Path | str,
    target: Literal["throughput_mbps", "bler"] = "throughput_mbps",
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str | None = None,
    checkpoint_dir: Path | str | None = None,
) -> dict[str, float]:
    """
    Main training function.

    Parameters
    ----------
    model_type : {"fc", "cnn"}
        Type of model to train
    csv_path : Path or str
        Path to summary.csv
    target : {"throughput_mbps", "bler"}
        Prediction target
    num_epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    learning_rate : float
        Learning rate
    device : str or None
        Device to use ("cuda", "cpu", or None for auto-detect)
    checkpoint_dir : Path or str or None
        Directory to save model checkpoints

    Returns
    -------
    dict
        Final evaluation metrics
    """
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print(f"Using device: {device}")

    # Create model
    use_channel_data = model_type == "cnn"
    if model_type == "fc":
        model = FCRegressor(input_dim=3, hidden_dims=(64, 32))
    elif model_type == "cnn":
        model = CNN1D(num_antennas=4, seq_len=4096, conv_channels=(16, 32, 64))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model.to(device)

    # Create dataloaders
    # Simple train/val split: use all data for training for now
    # In production, you'd want proper train/val/test split
    train_loader = create_dataloader(
        csv_path=csv_path,
        target=target,
        channel_data=use_channel_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Use same data for validation (for simplicity)
    val_loader = create_dataloader(
        csv_path=csv_path,
        target=target,
        channel_data=use_channel_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, use_channel_data
        )
        val_metrics = evaluate(model, val_loader, criterion, device, use_channel_data)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f} | "
            f"Val R2: {val_metrics['r2']:.4f}"
        )

        # Save best model
        if checkpoint_dir is not None and val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint_path = Path(checkpoint_dir) / f"best_{model_type}_{target}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )
            print(f"  -> Saved checkpoint to {checkpoint_path}")

    return val_metrics


def main():
    """
    Main entry point for training.

    Usage:
        python -m uav_acar_sionna.ml.train
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train ML models for UAV throughput prediction")
    parser.add_argument(
        "--model",
        type=str,
        choices=["fc", "cnn"],
        default="fc",
        help="Model type (fc or cnn)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/phase2_interference/summary.csv",
        help="Path to summary.csv",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["throughput_mbps", "bler"],
        default="throughput_mbps",
        help="Prediction target",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)",
    )

    args = parser.parse_args()

    print(f"Training {args.model} model on {args.target}")
    print(f"CSV: {args.csv}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print()

    metrics = train_model(
        model_type=args.model,
        csv_path=args.csv,
        target=args.target,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    print("\nFinal metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
