import torch
from torch import nn
from torch.nn import functional as F

from model_solar_lstm_v12 import OBSERVED_STEPS, SolarWindLagLSTMV12


ARCHITECTURE_NAME = "SolarWindLagLSTMUNetV12_1"
FILE_STEM = "solar_lag_lstm_unet_v12_1"


def _group_count(channels):
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = _group_count(out_channels)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, features):
        return self.layers(features)


class DownBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock2d(in_channels, out_channels),
        )

    def forward(self, features):
        return self.layers(features)


class UpBlock2d(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.fusion = ConvBlock2d(in_channels + skip_channels, out_channels)

    def forward(self, features, skip):
        features = F.interpolate(
            features,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.fusion(torch.cat([features, skip], dim=1))


class LiteUNetTokenEncoder(nn.Module):
    """Partial U-Net that stops at H/4 before source-token pooling."""

    def __init__(self, channels=(12, 16, 24, 40, 56), output_channels=128):
        super().__init__()
        if len(channels) != 5 or any(int(value) <= 0 for value in channels):
            raise ValueError("unet_channels must contain five positive values")
        c0, c1, c2, c3, c4 = (int(value) for value in channels)
        self.channels = (c0, c1, c2, c3, c4)
        self.input_block = ConvBlock2d(4, c0)
        self.down1 = DownBlock2d(c0, c1)
        self.down2 = DownBlock2d(c1, c2)
        self.down3 = DownBlock2d(c2, c3)
        self.bottleneck = DownBlock2d(c3, c4)
        self.up3 = UpBlock2d(c4, c3, c3)
        self.up2 = UpBlock2d(c3, c2, c2)
        self.output_projection = nn.Sequential(
            nn.Conv2d(c2, output_channels, 1, bias=False),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.GELU(),
        )

    def forward(self, features):
        skip0 = self.input_block(features)
        skip1 = self.down1(skip0)
        skip2 = self.down2(skip1)
        skip3 = self.down3(skip2)
        bottleneck = self.bottleneck(skip3)
        decoded = self.up3(bottleneck, skip3)
        decoded = self.up2(decoded, skip2)
        return self.output_projection(decoded)


class SolarWindLagLSTMUNetV12_1(SolarWindLagLSTMV12):
    """V12 AR/lag/LSTM back end with a lightweight U-Net token encoder."""

    def __init__(self, unet_channels=(12, 16, 24, 40, 56), **kwargs):
        image_size = int(kwargs.get("image_size", 64))
        if image_size % 16 != 0:
            raise ValueError("image_size must be divisible by 16")
        super().__init__(**kwargs)
        self.unet_channels = tuple(int(value) for value in unet_channels)

        # Remove the inherited CNN encoder while retaining its token projection,
        # LSTM, lag mixture, AR anchor, and correction heads.
        self.stem = nn.Identity()
        self.image_blocks = nn.ModuleList()
        self.unet_encoder = LiteUNetTokenEncoder(
            channels=self.unet_channels,
            output_channels=128,
        )

    def _encode_images(self, images):
        batch_size, time_steps = images.shape[:2]
        if time_steps != OBSERVED_STEPS:
            raise ValueError(f"expected {OBSERVED_STEPS} image steps")
        features = self._prepare_image_channels(images)
        features = features.reshape(
            batch_size * time_steps, 4, self.image_size, self.image_size
        )
        features = self.unet_encoder(features)
        height, width = features.shape[-2:]
        features = features.view(
            batch_size, time_steps, 128, height, width
        ).permute(0, 2, 1, 3, 4).contiguous()
        return self._project_frame_features(features, batch_size)
