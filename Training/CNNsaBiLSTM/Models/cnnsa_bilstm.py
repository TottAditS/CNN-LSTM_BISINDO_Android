"""
CNNsa-BiLSTM for MediaPipe keypoint sequences of shape (B, T, 126).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerFrameCNNBlock(nn.Module):
    """
    Two Conv1D layers over the 126-value keypoint vector of each frame, followed
    by a linear projection to feature_dim. The same weights are shared by all frames.
    """
    def __init__(self, input_size: int = 126, feature_dim: int = 256):
        super().__init__()
        # Input: (B * T, 1, input_size)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2) # 126 -> 63

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2) # 63 -> 31

        pool_out_len = input_size // 4  # 31 for input_size=126
        self.fc = nn.Linear(128 * pool_out_len, feature_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B * T, 1, input_size)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        return x


class SpatialAttention(nn.Module):
    """
    Channel attention over the feature dimension. A gate computed from the
    temporal average of the sequence reweights each of the feature_dim channels.
    """
    def __init__(self, feature_dim: int = 256, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, feature_dim // reduction, bias=False)
        self.fc2 = nn.Linear(feature_dim // reduction, feature_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, feature_dim)
        avg_pool = torch.mean(x, dim=1) # (B, feature_dim), global average over time
        w = F.relu(self.fc1(avg_pool))
        w = torch.sigmoid(self.fc2(w)) # (B, feature_dim)
        w = w.unsqueeze(dim=1) # (B, 1, feature_dim)
        return x * w


class TemporalAttention(nn.Module):
    """
    Scores each time step and returns the attention-weighted sum of the
    sequence (B, T, hidden_dim) as a context vector (B, hidden_dim).
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, hidden_dim)
        scores = self.w(x) # (B, T, 1)
        weights = F.softmax(scores, dim=1) # (B, T, 1)
        context = torch.sum(x * weights, dim=1) # (B, hidden_dim)
        return context, weights


class CNNsaBiLSTM(nn.Module):
    """
    Per-frame CNN -> channel attention -> BiLSTM -> temporal attention -> classifier.
    """
    def __init__(
        self,
        input_size: int = 126,
        feature_dim: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 32,
        dropout: float = 0.3,
        attn_reduction: int = 16,
    ):
        super().__init__()

        self.input_size = input_size
        self.feature_dim = feature_dim

        self.cnn_block = PerFrameCNNBlock(input_size=input_size, feature_dim=feature_dim)

        self.spatial_attn = SpatialAttention(feature_dim=feature_dim, reduction=attn_reduction)

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # BiLSTM output size is hidden_size * 2
        self.temporal_attn = TemporalAttention(hidden_dim=hidden_size * 2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, input_size)

        Returns:
            out: logits (B, num_classes)
            attn_weights: temporal attention weights (B, T, 1)
        """
        B, T, F_size = x.shape
        assert F_size == self.input_size, f"Input feature size mismatch. Expected {self.input_size}, got {F_size}"

        # Run the CNN on every frame as a 1-channel sequence
        x_flat = x.view(B * T, 1, F_size)
        features_flat = self.cnn_block(x_flat) # (B * T, feature_dim)
        features = features_flat.view(B, T, self.feature_dim) # (B, T, feature_dim)

        features = self.spatial_attn(features) # (B, T, feature_dim)

        lstm_out, _ = self.lstm(features) # (B, T, hidden_size * 2)

        context, attn_weights = self.temporal_attn(lstm_out) # context: (B, hidden_size * 2)

        out = self.classifier(context) # (B, num_classes)

        return out, attn_weights


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNsaBiLSTM().to(device)

    dummy_input = torch.randn(8, 30, 126).to(device)

    out, attn = model(dummy_input)
    print("Model output shape  :", out.shape)  # (8, 32)
    print("Attention shape     :", attn.shape) # (8, 30, 1)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {total_params:,}")
