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
from model_tile_transformer import make_wind_features, sinusoidal_time_encoding
from probabilistic import LowRankStudentTHead


OBSERVED_STEPS = 20
FORECAST_STEPS = 12
SPATIAL_SIZE = 4


def channel_norm(channels):
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ObserverAlignedCEA(nn.Module):
    """Approximate equal-area reprojection for centered, north-up disk PNGs."""

    def __init__(self, image_size, radius_fraction):
        super().__init__()
        longitude_margin = math.pi / (2.0 * image_size)
        longitude = torch.linspace(
            -math.pi / 2.0 + longitude_margin,
            math.pi / 2.0 - longitude_margin,
            image_size,
        )
        sin_latitude = torch.linspace(
            -1.0 + 1.0 / image_size,
            1.0 - 1.0 / image_size,
            image_size,
        )
        sin_latitude, longitude = torch.meshgrid(
            sin_latitude, longitude, indexing="ij"
        )
        cos_latitude = torch.sqrt((1.0 - sin_latitude.square()).clamp_min(0.0))

        normalized_radius = 2.0 * radius_fraction * image_size / (image_size - 1)
        disk_x = normalized_radius * cos_latitude * torch.sin(longitude)
        disk_y = normalized_radius * sin_latitude
        mu = cos_latitude * torch.cos(longitude)
        latitude = torch.asin(sin_latitude)

        self.register_buffer(
            "sampling_grid",
            torch.stack([disk_x, disk_y], dim=-1).unsqueeze(0),
            persistent=False,
        )
        coordinates = torch.stack(
            [
                longitude / (math.pi / 2.0),
                latitude.abs() / (math.pi / 2.0),
                mu,
            ],
            dim=0,
        )
        self.register_buffer("coordinates", coordinates.unsqueeze(0), persistent=False)
        self.register_buffer(
            "disk_mask",
            make_solar_disk_mask(
                image_size, image_size, (0.5, 0.5), radius_fraction
            ),
            persistent=False,
        )

    def forward(self, images):
        batch, steps, channels, height, width = images.shape
        disk_mask = self.disk_mask.to(device=images.device, dtype=images.dtype)
        flat = (images * disk_mask).reshape(batch * steps, channels, height, width)
        grid = self.sampling_grid.to(device=images.device, dtype=images.dtype)
        grid = grid.expand(batch * steps, -1, -1, -1)
        projected = F.grid_sample(
            flat,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        coordinates = self.coordinates.to(device=images.device, dtype=images.dtype)
        coordinates = coordinates.expand(batch * steps, -1, -1, -1)
        return projected, coordinates


class MultiScaleSolarBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if out_channels % 4 != 0:
            raise ValueError("out_channels must be divisible by four")
        branch_channels = out_channels // 4

        self.point = nn.Conv2d(in_channels, branch_channels, kernel_size=1)
        self.local_branches = nn.ModuleList()
        for dilation in (1, 2, 4):
            self.local_branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        groups=in_channels,
                    ),
                    channel_norm(in_channels),
                    nn.GELU(),
                    nn.Conv2d(in_channels, branch_channels, kernel_size=1),
                )
            )
        self.output_norm = channel_norm(out_channels)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, features):
        branches = [self.point(features)]
        branches.extend(branch(features) for branch in self.local_branches)
        combined = self.output_norm(torch.cat(branches, dim=1))
        return F.gelu(combined + self.residual(features))


class DualPolarityDownsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            channel_norm(channels),
            nn.GELU(),
        )

    def forward(self, features):
        average = F.avg_pool2d(features, kernel_size=2, stride=2)
        maximum = F.max_pool2d(features, kernel_size=2, stride=2)
        minimum = -F.max_pool2d(-features, kernel_size=2, stride=2)
        return self.projection(torch.cat([average, maximum, minimum], dim=1))


class SolarGeometryEncoder(nn.Module):
    def __init__(self, image_size, radius_fraction, output_channels):
        super().__init__()
        self.reprojection = ObserverAlignedCEA(image_size, radius_fraction)
        self.stem = nn.Sequential(
            nn.Conv2d(7, 24, kernel_size=5, stride=2, padding=2),
            channel_norm(24),
            nn.GELU(),
        )
        self.block_1 = MultiScaleSolarBlock(24, 48)
        self.downsample_1 = DualPolarityDownsample(48)
        self.block_2 = MultiScaleSolarBlock(48, output_channels)
        self.downsample_2 = DualPolarityDownsample(output_channels)

    def forward(self, images):
        batch, steps = images.shape[:2]
        projected, coordinates = self.reprojection(images)
        reliability = coordinates[:, 2:3]
        brightness = projected * reliability
        darkness = (1.0 - projected) * reliability
        features = torch.cat([brightness, darkness, coordinates], dim=1)
        features = self.stem(features)
        features = self.downsample_1(self.block_1(features))
        features = self.downsample_2(self.block_2(features))
        features = F.adaptive_avg_pool2d(features, (SPATIAL_SIZE, SPATIAL_SIZE))
        features = features.flatten(2).transpose(1, 2)
        return features.reshape(batch, steps, SPATIAL_SIZE**2, -1)


class SolarWindProbabilisticTransformer(nn.Module):
    def __init__(
        self,
        image_size=IMAGE_SIZE,
        apply_solar_disk_mask=SOLAR_DISK_MASK,
        solar_disk_center_fraction=SOLAR_DISK_CENTER_FRACTION,
        solar_disk_radius_fraction=SOLAR_DISK_RADIUS_FRACTION,
        d_model=96,
        wind_dim=24,
        nhead=8,
        encoder_layers=1,
        ff_dim=192,
        dropout=0.15,
        distribution_rank=3,
        use_images=True,
        baseline_slope=None,
        baseline_intercept=None,
        baseline_residual_scale=None,
    ):
        super().__init__()
        if not apply_solar_disk_mask:
            raise ValueError("the CEA model requires solar disk masking")
        if solar_disk_center_fraction != (0.5, 0.5):
            raise ValueError("the CEA model requires a centered solar disk")
        if d_model <= wind_dim or d_model % nhead != 0:
            raise ValueError("invalid d_model, wind_dim, or nhead combination")

        self.use_images = use_images
        self.image_feature_dim = d_model - wind_dim
        self.image_encoder = SolarGeometryEncoder(
            image_size, solar_disk_radius_fraction, self.image_feature_dim
        )
        self.wind_projection = nn.Sequential(
            nn.Linear(4, wind_dim),
            nn.GELU(),
            nn.Linear(wind_dim, wind_dim),
            nn.LayerNorm(wind_dim),
        )
        self.memory_norm = nn.LayerNorm(d_model)
        self.observed_position = nn.Parameter(
            torch.randn(1, OBSERVED_STEPS, 1, d_model) * 0.1
        )
        self.spatial_position = nn.Parameter(
            torch.randn(1, 1, SPATIAL_SIZE**2, d_model) * 0.05
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
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
        self.distribution_head = LowRankStudentTHead(
            d_model,
            FORECAST_STEPS,
            rank=distribution_rank,
            baseline_residual_scale=baseline_residual_scale,
            dropout=dropout,
        )

        if baseline_slope is None:
            baseline_slope = torch.ones(FORECAST_STEPS)
        if baseline_intercept is None:
            baseline_intercept = torch.zeros(FORECAST_STEPS)
        self.register_buffer(
            "baseline_slope",
            torch.as_tensor(baseline_slope, dtype=torch.float32).view(1, -1),
        )
        self.register_buffer(
            "baseline_intercept",
            torch.as_tensor(baseline_intercept, dtype=torch.float32).view(1, -1),
        )

    def linear_baseline(self, wind):
        return wind[:, -1:] * self.baseline_slope + self.baseline_intercept

    def forward(self, images, wind, return_distribution=False):
        batch = images.size(0)
        wind_tokens = self.wind_projection(make_wind_features(wind))
        if self.use_images:
            image_tokens = self.image_encoder(images)
        else:
            image_tokens = wind_tokens.new_zeros(
                batch, OBSERVED_STEPS, SPATIAL_SIZE**2, self.image_feature_dim
            )

        repeated_wind = wind_tokens.unsqueeze(2).expand(-1, -1, SPATIAL_SIZE**2, -1)
        spatial_memory = torch.cat([image_tokens, repeated_wind], dim=-1)
        spatial_memory = self.memory_norm(
            spatial_memory + self.observed_position + self.spatial_position
        )

        temporal_memory = self.temporal_encoder(spatial_memory.mean(dim=2))
        spatial_memory = spatial_memory.flatten(1, 2)
        memory = torch.cat([temporal_memory, spatial_memory], dim=1)

        queries = self.future_queries + self.future_time_encoding
        queries = queries.expand(batch, -1, -1)
        attended, _ = self.query_attention(
            queries, memory, memory, need_weights=False
        )
        queries = self.query_norm(queries + self.dropout(attended))
        queries = self.query_ffn_norm(
            queries + self.dropout(self.query_ffn(queries))
        )

        residual, diagonal_scale, factors, degrees_of_freedom = (
            self.distribution_head(queries)
        )
        prediction = self.linear_baseline(wind) + residual
        if return_distribution:
            return prediction, diagonal_scale, factors, degrees_of_freedom
        return prediction
