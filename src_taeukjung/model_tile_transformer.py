import math

import torch
from torch import nn
from torch.nn import functional as F

from config import (
    IMAGE_SIZE,
    SOLAR_DISK_CENTER_FRACTION,
    SOLAR_DISK_MASK,
    SOLAR_DISK_RADIUS_FRACTION,
)
from model import make_solar_disk_mask


OBSERVED_STEPS = 20
FORECAST_STEPS = 12
IMAGE_CHANNELS = 2
TILE_STATISTICS = 4


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


def masked_tile_statistics(images, mask, grid_size):
    """Return mean, minimum, maximum, and std for each masked image tile."""
    if images.ndim != 5:
        raise ValueError(f"expected images shaped (B,T,C,H,W), got {images.shape}")

    batch, steps, channels, height, width = images.shape
    if height % grid_size != 0 or width % grid_size != 0:
        raise ValueError(
            f"image size {(height, width)} must be divisible by grid size {grid_size}"
        )

    kernel = (height // grid_size, width // grid_size)
    flat = images.reshape(batch * steps * channels, 1, height, width)
    flat_mask = mask.reshape(1, 1, height, width).to(
        device=images.device, dtype=images.dtype
    )
    valid_pixels = flat_mask > 0.5

    coverage = F.avg_pool2d(flat_mask, kernel_size=kernel, stride=kernel)
    denominator = coverage.clamp_min(1.0 / (kernel[0] * kernel[1]))
    mean = F.avg_pool2d(flat * flat_mask, kernel, kernel) / denominator
    second_moment = F.avg_pool2d(flat.square() * flat_mask, kernel, kernel)
    second_moment = second_moment / denominator
    variance = (second_moment - mean.square()).clamp_min(0.0)
    std = torch.sqrt(variance.clamp_min(1e-8))

    lowest = torch.finfo(images.dtype).min
    maximum = F.max_pool2d(
        flat.masked_fill(~valid_pixels, lowest), kernel, kernel
    )
    minimum = -F.max_pool2d(
        (-flat).masked_fill(~valid_pixels, lowest), kernel, kernel
    )

    valid_tiles = coverage > 0
    mean = torch.where(valid_tiles, mean, 0.0)
    minimum = torch.where(valid_tiles, minimum, 0.0)
    maximum = torch.where(valid_tiles, maximum, 0.0)
    std = torch.where(valid_tiles, std, 0.0)

    statistics = torch.stack([mean, minimum, maximum, std], dim=2)
    return statistics.reshape(batch, steps, channels * TILE_STATISTICS * grid_size**2)


class SolarWindTileTransformer(nn.Module):
    def __init__(
        self,
        image_size=IMAGE_SIZE,
        apply_solar_disk_mask=SOLAR_DISK_MASK,
        solar_disk_center_fraction=SOLAR_DISK_CENTER_FRACTION,
        solar_disk_radius_fraction=SOLAR_DISK_RADIUS_FRACTION,
        tile_grid_size=8,
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
        if image_size % tile_grid_size != 0:
            raise ValueError(
                f"image_size {image_size} must be divisible by tile_grid_size "
                f"{tile_grid_size}"
            )
        if d_model <= wind_dim:
            raise ValueError("d_model must be larger than wind_dim")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.apply_solar_disk_mask = apply_solar_disk_mask
        self.solar_disk_center_fraction = solar_disk_center_fraction
        self.solar_disk_radius_fraction = solar_disk_radius_fraction
        self.tile_grid_size = tile_grid_size
        self.use_images = use_images

        if apply_solar_disk_mask:
            mask = make_solar_disk_mask(
                image_size,
                image_size,
                solar_disk_center_fraction,
                solar_disk_radius_fraction,
            )
        else:
            mask = torch.ones(1, 1, 1, image_size, image_size)
        self.register_buffer("solar_disk_mask", mask, persistent=False)

        image_dim = d_model - wind_dim
        tile_feature_dim = IMAGE_CHANNELS * TILE_STATISTICS * tile_grid_size**2
        image_hidden_dim = max(image_dim * 2, 128)
        self.image_projection = nn.Sequential(
            nn.Linear(tile_feature_dim, image_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(image_hidden_dim, image_dim),
            nn.LayerNorm(image_dim),
        )
        self.wind_projection = nn.Sequential(
            nn.Linear(4, wind_dim),
            nn.GELU(),
            nn.Linear(wind_dim, wind_dim),
            nn.LayerNorm(wind_dim),
        )

        self.observed_position = nn.Parameter(
            torch.randn(1, OBSERVED_STEPS, d_model) * 0.1
        )
        self.fusion_norm = nn.LayerNorm(d_model)

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
            enable_nested_tensor=False,
        )

        future_positions = torch.arange(1, FORECAST_STEPS + 1)
        self.register_buffer(
            "future_time_encoding",
            sinusoidal_time_encoding(future_positions, d_model),
            persistent=False,
        )
        self.future_queries = nn.Parameter(
            torch.randn(1, FORECAST_STEPS, d_model) * 0.02
        )
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
        nn.init.normal_(self.residual_head[-1].weight, std=1e-3)
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

    def _image_mask(self, images):
        if self.solar_disk_mask.shape[-2:] == images.shape[-2:]:
            return self.solar_disk_mask
        if not self.apply_solar_disk_mask:
            return images.new_ones(1, 1, 1, images.shape[-2], images.shape[-1])
        return make_solar_disk_mask(
            images.shape[-2],
            images.shape[-1],
            self.solar_disk_center_fraction,
            self.solar_disk_radius_fraction,
            device=images.device,
            dtype=images.dtype,
        )

    def _encode_images(self, images, wind_tokens):
        if not self.use_images:
            return wind_tokens.new_zeros(
                images.size(0), OBSERVED_STEPS, self.image_projection[-1].normalized_shape[0]
            )

        statistics = masked_tile_statistics(
            images, self._image_mask(images), self.tile_grid_size
        )
        return self.image_projection(statistics)

    def linear_baseline(self, wind):
        return wind[:, -1:] * self.baseline_slope + self.baseline_intercept

    def forward(self, images, wind):
        if wind.shape[1] != OBSERVED_STEPS:
            raise ValueError(f"expected {OBSERVED_STEPS} wind steps, got {wind.shape[1]}")
        if images.shape[1] != OBSERVED_STEPS:
            raise ValueError(
                f"expected {OBSERVED_STEPS} image steps, got {images.shape[1]}"
            )

        wind_tokens = self.wind_projection(make_wind_features(wind))
        image_tokens = self._encode_images(images, wind_tokens)
        tokens = torch.cat([image_tokens, wind_tokens], dim=-1)
        memory = self.encoder(self.fusion_norm(tokens + self.observed_position))

        queries = self.future_queries + self.future_time_encoding
        queries = queries.expand(images.size(0), -1, -1)
        attended, _ = self.query_attention(
            queries, memory, memory, need_weights=False
        )
        queries = self.query_norm(queries + self.dropout(attended))
        queries = self.query_ffn_norm(
            queries + self.dropout(self.query_ffn(queries))
        )

        residual = self.residual_head(queries).squeeze(-1)
        return self.linear_baseline(wind) + residual
