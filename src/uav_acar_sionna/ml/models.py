"""
PyTorch models for throughput and BLER prediction.
"""

import torch
import torch.nn as nn


class FCRegressor(nn.Module):
    """
    Simple fully-connected regression network.

    Takes tabular features (distance_m, snr_db, sinr_db) and predicts
    throughput_mbps or bler.

    Architecture:
        Input (3) -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> Linear(1)

    Parameters
    ----------
    input_dim : int
        Number of input features (default: 3 for distance_m, snr_db, sinr_db)
    hidden_dims : tuple of int
        Sizes of hidden layers (default: (64, 32))
    """

    def __init__(self, input_dim: int = 3, hidden_dims: tuple[int, ...] = (64, 32)):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch_size, input_dim)

        Returns
        -------
        torch.Tensor
            Predictions of shape (batch_size, 1)
        """
        return self.network(x)


class CNN1D(nn.Module):
    """
    1D CNN for channel-based throughput prediction.

    Processes complex channel matrices (h_real, h_imag) from NPZ files
    to predict throughput_mbps or bler.

    Architecture:
        - Input: complex channel tensor (4, 4096)
        - Stack real and imag as 8 channels
        - Conv1D layers to extract features
        - Global average pooling
        - FC layers for regression

    Parameters
    ----------
    num_antennas : int
        Number of antennas (default: 4)
    seq_len : int
        Sequence length of channel samples (default: 4096)
    conv_channels : tuple of int
        Number of channels in each conv layer (default: (16, 32, 64))
    fc_hidden_dim : int
        Hidden dimension of final FC layer (default: 64)
    """

    def __init__(
        self,
        num_antennas: int = 4,
        seq_len: int = 4096,
        conv_channels: tuple[int, ...] = (16, 32, 64),
        fc_hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_antennas = num_antennas
        self.seq_len = seq_len
        self.conv_channels = conv_channels
        self.fc_hidden_dim = fc_hidden_dim

        # Input channels: num_antennas * 2 (real and imag stacked)
        in_channels = num_antennas * 2

        # Build conv layers
        conv_layers = []
        prev_channels = in_channels
        for out_channels in conv_channels:
            conv_layers.append(
                nn.Conv1d(
                    prev_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool1d(kernel_size=2))
            prev_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # FC layers
        self.fc = nn.Sequential(
            nn.Linear(conv_channels[-1], fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, 1),
        )

    def forward(self, h_complex: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        h_complex : torch.Tensor
            Complex channel tensor of shape (batch_size, num_antennas, seq_len)

        Returns
        -------
        torch.Tensor
            Predictions of shape (batch_size, 1)
        """
        batch_size = h_complex.shape[0]

        # Stack real and imag as separate channels
        h_real = h_complex.real  # (batch, num_antennas, seq_len)
        h_imag = h_complex.imag  # (batch, num_antennas, seq_len)
        h_stacked = torch.cat([h_real, h_imag], dim=1)  # (batch, num_antennas*2, seq_len)

        # Conv layers
        x = self.conv(h_stacked)  # (batch, conv_channels[-1], seq_len')

        # Global average pooling
        x = self.gap(x)  # (batch, conv_channels[-1], 1)
        x = x.squeeze(-1)  # (batch, conv_channels[-1])

        # FC layers
        out = self.fc(x)  # (batch, 1)

        return out
