import torch
from torch import nn
from torch.nn import functional as F

from config import (
    IMAGE_SIZE,
    SOLAR_DISK_CENTER_FRACTION,
    SOLAR_DISK_MASK,
    SOLAR_DISK_RADIUS_FRACTION,
)
from model_solar_probabilistic import (
    FORECAST_STEPS,
    OBSERVED_STEPS,
    DualPolarityDownsample,
    MultiScaleSolarBlock,
    ObserverAlignedCEA,
    channel_norm,
)
from model_tile_transformer import make_wind_features, sinusoidal_time_encoding
from probabilistic import LowRankStudentTHead


class SolarGeometryEncoderV2(nn.Module):
    """CEA encoder that preserves raw limb intensity and more longitude bins."""

    def __init__(
        self,
        image_size,
        radius_fraction,
        output_channels,
        spatial_height=4,
        spatial_width=8,
    ):
        super().__init__()
        if spatial_height <= 0 or spatial_width <= 0:
            raise ValueError("spatial dimensions must be positive")

        self.spatial_height = spatial_height
        self.spatial_width = spatial_width
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

        # mu remains an independent coordinate channel. The CNN can learn how much
        # to trust the limb without irreversibly erasing bright or dark structures.
        brightness = projected
        darkness = 1.0 - projected
        features = torch.cat([brightness, darkness, coordinates], dim=1)
        features = self.stem(features)
        features = self.downsample_1(self.block_1(features))
        features = self.downsample_2(self.block_2(features))
        features = F.adaptive_avg_pool2d(
            features, (self.spatial_height, self.spatial_width)
        )
        features = features.flatten(2).transpose(1, 2)
        return features.reshape(
            batch,
            steps,
            self.spatial_height * self.spatial_width,
            -1,
        )


class SolarWindProbabilisticTransformerV2(nn.Module):
    """Version 2 with unsuppressed CEA intensity and a rectangular token grid."""

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        apply_solar_disk_mask=SOLAR_DISK_MASK,
        solar_disk_center_fraction=SOLAR_DISK_CENTER_FRACTION,
        solar_disk_radius_fraction=SOLAR_DISK_RADIUS_FRACTION,
        spatial_height=4,
        spatial_width=8,
        d_model=96,
        wind_dim=24,
        nhead=8,
        encoder_layers=1,
        ff_dim=192,
        dropout=0.25,
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
        if spatial_height <= 0 or spatial_width <= 0:
            raise ValueError("spatial dimensions must be positive")

        self.use_images = use_images
        self.spatial_height = spatial_height
        self.spatial_width = spatial_width
        self.spatial_token_count = spatial_height * spatial_width
        self.image_feature_dim = d_model - wind_dim
        self.image_encoder = SolarGeometryEncoderV2(
            image_size,
            solar_disk_radius_fraction,
            self.image_feature_dim,
            spatial_height=spatial_height,
            spatial_width=spatial_width,
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
            torch.randn(1, 1, self.spatial_token_count, d_model) * 0.05
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
                batch,
                OBSERVED_STEPS,
                self.spatial_token_count,
                self.image_feature_dim,
            )

        repeated_wind = wind_tokens.unsqueeze(2).expand(
            -1, -1, self.spatial_token_count, -1
        )
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
