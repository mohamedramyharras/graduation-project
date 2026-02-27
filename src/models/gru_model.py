"""
GRU architecture for EMG-to-Force prediction.

Architecture: GRU(hidden, relu) -> Dropout -> Dense(dense, relu) -> Dropout -> Dense(1)
Based on Ghorbani et al. 2023, adapted for 2-channel grip force prediction.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class GRUForcePredictor(nn.Module):
    """
    GRU-based force predictor.

    Parameters
    ----------
    input_size : int
        Number of EMG channels (default: 2 after MRMR selection).
    hidden_size : int
        GRU hidden units.
    dense_size : int
        Dense layer units.
    output_size : int
        Output dimension (default: 1 for total grip force).
    dropout : float
        Dropout probability after GRU and dense layers (default: 0.0).
    """

    def __init__(self, input_size=2, hidden_size=50, dense_size=100, output_size=1, dropout=0.0, num_layers=1, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        # For bidirectional, GRU output is hidden_size * 2
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(gru_output_size, dense_size)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(dense_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)                 # (batch, seq_len, hidden*D)
        # For bidirectional: take last forward + first backward hidden state
        if self.bidirectional:
            hidden_size = gru_out.shape[2] // 2
            # Forward direction: last time step
            fwd_last = gru_out[:, -1, :hidden_size]
            # Backward direction: first time step
            bwd_first = gru_out[:, 0, hidden_size:]
            last = torch.cat([fwd_last, bwd_first], dim=1)
        else:
            last = gru_out[:, -1, :]              # (batch, hidden)
        last = self.relu1(last)
        last = self.drop1(last)
        x = self.relu2(self.fc1(last))            # (batch, dense)
        x = self.drop2(x)
        return self.fc_out(x)                     # (batch, output)


class AttentionGRUPredictor(nn.Module):
    """GRU with temporal attention for EMG-to-Force prediction.

    Instead of using only the last hidden state, this model computes
    attention weights over all time steps and produces a weighted context
    vector.  This allows the model to focus on the most informative parts
    of the input sequence.

    Architecture:
        GRU -> Attention(all timesteps) -> Dropout -> Dense(relu) -> Dropout -> Dense(1)
    """

    def __init__(self, input_size=12, hidden_size=64, dense_size=64,
                 output_size=1, dropout=0.2, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Attention: score each time step
        self.attn_fc = nn.Linear(hidden_size, 1)
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, dense_size)
        self.relu = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(dense_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)                     # (B, T, H)
        attn_scores = self.attn_fc(gru_out)           # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(gru_out * attn_weights, dim=1)  # (B, H)
        context = self.drop1(context)
        x = self.relu(self.fc1(context))
        x = self.drop2(x)
        return self.fc_out(x)


class SeqDataset(Dataset):
    """PyTorch Dataset for EMG-Force sequence pairs."""

    def __init__(self, X, y):
        """
        Parameters
        ----------
        X : ndarray of shape (n_samples, seq_len, n_channels)
        y : ndarray of shape (n_samples,) or (n_samples, output_dim)
        """
        self.X = torch.FloatTensor(X)
        if y.ndim == 1:
            self.y = torch.FloatTensor(y).unsqueeze(1)
        else:
            self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def count_parameters(model):
    """Count total and trainable parameters in a PyTorch model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
