import math

import torch
from torch import nn
from torch.nn import functional as F

from config import (
    IMAGE_SIZE,
    SOLAR_DISK_CENTER_FRACTION,
    SOLAR_DISK_MASK,
    SOLAR_DISK_RADIUS_FRACTION,
    SPATIAL_FEATURE_SIZE,
)
from model import Inception3D, make_solar_disk_mask, spatial_block_average


OBSERVED_STEPS = 20
FORECAST_STEPS = 12


def sinusoidal_time_encoding(positions, d_model):
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even, got {d_model}")

    positions = torch.as_tensor(positions, dtype=torch.float32).view(-1, 1)
    frequencies = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    angles = positions * frequencies
    encoding = torch.zeros(len(positions), d_model, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles)
    return encoding.unsqueeze(0)


def make_wind_features(wind):
    delta = torch.zeros_like(wind)
    delta[:, 1:] = wind[:, 1:] - wind[:, :-1]

    acceleration = torch.zeros_like(wind)
    acceleration[:, 1:] = delta[:, 1:] - delta[:, :-1]

    padded = F.pad(wind.unsqueeze(1), (2, 0), mode="replicate")
    rolling_mean = F.avg_pool1d(padded, kernel_size=3, stride=1).squeeze(1)
    return torch.stack([wind, delta, acceleration, rolling_mean], dim=-1)


class TemporalMixingBlock(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=1, groups=d_model
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens):
        mixed = tokens.transpose(1, 2)
        mixed = self.depthwise(mixed)
        mixed = F.gelu(mixed)
        mixed = self.pointwise(mixed).transpose(1, 2)
        return self.norm(tokens + self.dropout(mixed))


class SolarWindTransformer(nn.Module):
    def __init__(
        self,
        image_size=IMAGE_SIZE,
        apply_solar_disk_mask=SOLAR_DISK_MASK,
        solar_disk_center_fraction=SOLAR_DISK_CENTER_FRACTION,
        solar_disk_radius_fraction=SOLAR_DISK_RADIUS_FRACTION,
        spatial_feature_size=SPATIAL_FEATURE_SIZE,
        d_model=128,
        wind_dim=32,
        nhead=8,
        encoder_layers=2,
        ff_dim=256,
        dropout=0.1,
        use_images=True,
        baseline_slope=None,
        baseline_intercept=None,
    ):
        super().__init__()
        if d_model <= wind_dim:
            raise ValueError("d_model must be larger than wind_dim")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.apply_solar_disk_mask = apply_solar_disk_mask
        self.solar_disk_center_fraction = solar_disk_center_fraction
        self.solar_disk_radius_fraction = solar_disk_radius_fraction
        self.spatial_feature_size = spatial_feature_size
        self.use_images = use_images

        if apply_solar_disk_mask:
            mask = make_solar_disk_mask(
                image_size,
                image_size,
                solar_disk_center_fraction,
                solar_disk_radius_fraction,
            )
        else:
            mask = torch.ones(1, 1, 1, 1, 1)
        self.register_buffer("solar_disk_mask", mask, persistent=False)

        self.stem = nn.Sequential(
            nn.Conv3d(2, 32, (1, 5, 5), padding=(0, 2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        blocks = []
        in_channels = 32
        for _ in range(3):
            blocks.extend(
                [
                    Inception3D(in_channels, 32),
                    nn.MaxPool3d(
                        (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
                    ),
                ]
            )
            in_channels = 128
        self.image_encoder = nn.Sequential(*blocks)

        image_dim = d_model - wind_dim
        self.image_projection = nn.Sequential(
            nn.Linear(128 * spatial_feature_size * spatial_feature_size, image_dim),
            nn.LayerNorm(image_dim),
        )
        self.wind_projection = nn.Sequential(
            nn.Linear(4, wind_dim),
            nn.GELU(),
            nn.Linear(wind_dim, wind_dim),
            nn.LayerNorm(wind_dim),
        )

        self.fusion_norm = nn.LayerNorm(d_model)
        self.temporal_mixer = TemporalMixingBlock(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        observed_positions = torch.arange(-OBSERVED_STEPS + 1, 1)
        future_positions = torch.arange(1, FORECAST_STEPS + 1)
        self.register_buffer(
            "observed_time_encoding",
            sinusoidal_time_encoding(observed_positions, d_model),
            persistent=False,
        )
        self.register_buffer(
            "future_time_encoding",
            sinusoidal_time_encoding(future_positions, d_model),
            persistent=False,
        )
        self.position_scale = nn.Parameter(torch.tensor(0.25))

        self.query_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.query_norm = nn.LayerNorm(d_model)
        self.query_ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.query_ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.residual_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

        if baseline_slope is None:
            baseline_slope = torch.ones(FORECAST_STEPS)
        if baseline_intercept is None:
            baseline_intercept = torch.zeros(FORECAST_STEPS)
        if len(baseline_slope) != FORECAST_STEPS:
            raise ValueError(f"baseline_slope must have {FORECAST_STEPS} values")
        if len(baseline_intercept) != FORECAST_STEPS:
            raise ValueError(f"baseline_intercept must have {FORECAST_STEPS} values")
        self.register_buffer(
            "baseline_slope",
            torch.as_tensor(baseline_slope, dtype=torch.float32).view(1, -1),
        )
        self.register_buffer(
            "baseline_intercept",
            torch.as_tensor(baseline_intercept, dtype=torch.float32).view(1, -1),
        )

    def _mask_images(self, images):
        if not self.apply_solar_disk_mask:
            return images
        if self.solar_disk_mask.shape[-2:] == images.shape[-2:]:
            mask = self.solar_disk_mask.to(device=images.device, dtype=images.dtype)
        else:
            mask = make_solar_disk_mask(
                images.shape[-2],
                images.shape[-1],
                self.solar_disk_center_fraction,
                self.solar_disk_radius_fraction,
                device=images.device,
                dtype=images.dtype,
            )
        return images * mask

    def _encode_images(self, images, wind_tokens):
        if not self.use_images:
            return wind_tokens.new_zeros(
                images.size(0), OBSERVED_STEPS, self.image_projection[0].out_features
            )

        images = self._mask_images(images)
        features = images.permute(0, 2, 1, 3, 4).contiguous()
        features = self.image_encoder(self.stem(features))
        features = spatial_block_average(features, self.spatial_feature_size)
        features = features.permute(0, 2, 1, 3, 4).flatten(2)
        return self.image_projection(features)

    def forward(self, images, wind):
        if wind.shape[1] != OBSERVED_STEPS:
            raise ValueError(f"expected {OBSERVED_STEPS} wind steps, got {wind.shape[1]}")

        wind_tokens = self.wind_projection(make_wind_features(wind))
        image_tokens = self._encode_images(images, wind_tokens)

        tokens = self.fusion_norm(torch.cat([image_tokens, wind_tokens], dim=-1))
        tokens = self.temporal_mixer(tokens)
        tokens = tokens + self.position_scale * self.observed_time_encoding
        memory = self.encoder(tokens)

        queries = self.future_time_encoding.expand(images.size(0), -1, -1)
        attended, _ = self.query_attention(
            queries, memory, memory, need_weights=False
        )
        queries = self.query_norm(queries + self.dropout(attended))
        queries = self.query_ffn_norm(
            queries + self.dropout(self.query_ffn(queries))
        )

        residual = self.residual_head(queries).squeeze(-1)
        baseline = wind[:, -1:] * self.baseline_slope + self.baseline_intercept
        return baseline + residual
