import torch
import torch.nn as nn


class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x):
        return x + self.block(x)


class TemporalTinyRestorationCNN(nn.Module):
    """
    Small causal temporal image-restoration frontend.

    The final projection is zero-initialized so the network starts as an identity
    mapping. This keeps the initial restored accuracy close to the nonideal
    baseline and makes short pilot runs easier to interpret.
    """

    def __init__(self, in_channels=3, hidden_channels=16, num_blocks=3, history_frames=1):
        super().__init__()
        if history_frames <= 0:
            raise ValueError(f"history_frames must be positive, got {history_frames}")
        self.in_channels = in_channels
        self.history_frames = history_frames
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels * history_frames, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualConvBlock(hidden_channels) for _ in range(num_blocks)])
        self.head = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        current_frame, stacked_frames = self._reshape_input(x)
        features = self.stem(stacked_frames)
        features = self.blocks(features)
        correction = self.head(features)
        return current_frame + correction

    def _reshape_input(self, x):
        if x.ndim == 5:
            batch_size, num_frames, num_channels, height, width = x.shape
            if num_frames != self.history_frames:
                raise ValueError(
                    f"Expected {self.history_frames} frames, got {num_frames}"
                )
            if num_channels != self.in_channels:
                raise ValueError(
                    f"Expected {self.in_channels} channels, got {num_channels}"
                )
            current_frame = x[:, -1, :, :, :]
            stacked_frames = x.reshape(batch_size, num_frames * num_channels, height, width)
            return current_frame, stacked_frames

        if x.ndim == 4:
            if x.shape[1] == self.in_channels and self.history_frames == 1:
                return x, x
            expected_channels = self.in_channels * self.history_frames
            if x.shape[1] != expected_channels:
                raise ValueError(
                    f"Expected flattened temporal input with {expected_channels} channels, got {x.shape[1]}"
                )
            current_frame = x[:, -self.in_channels:, :, :]
            return current_frame, x

        raise ValueError(f"Expected 4D or 5D input, got rank {x.ndim}")


class TinyRestorationCNN(TemporalTinyRestorationCNN):
    def __init__(self, in_channels=3, hidden_channels=16, num_blocks=3, history_frames=1):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            history_frames=history_frames,
        )
