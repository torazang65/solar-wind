import torch
from torch import nn
from torch.nn import functional as F

from config import (
    IMAGE_SIZE,
    SOLAR_CEA_RADIUS_FRACTION,
    SOLAR_DISK_CENTER_FRACTION,
    SOLAR_DISK_MASK,
    SOLAR_DISK_RADIUS_FRACTION,
)
from model_solar_probabilistic import FORECAST_STEPS, OBSERVED_STEPS, ObserverAlignedCEA
from model_tile_transformer import make_wind_features, sinusoidal_time_encoding


class CoronalHoleFeatureExtractor(nn.Module):
    """Extract fixed CEA cell statistics that preserve dark-region geometry."""

    def __init__(
        self,
        image_size,
        mask_radius_fraction,
        cea_radius_fraction,
        latitude_bins=4,
        longitude_bins=8,
        dark_thresholds=(0.15, 0.30),
        quantiles=(0.10, 0.25, 0.50),
    ):
        super().__init__()
        if latitude_bins <= 0 or longitude_bins <= 0:
            raise ValueError("latitude and longitude bin counts must be positive")
        self.latitude_bins = latitude_bins
        self.longitude_bins = longitude_bins
        self.cell_count = latitude_bins * longitude_bins
        self.dark_thresholds = tuple(float(value) for value in dark_thresholds)
        self.quantiles = tuple(float(value) for value in quantiles)
        self.reprojection = ObserverAlignedCEA(
            image_size,
            cea_radius_fraction,
            mask_radius_fraction=mask_radius_fraction,
        )

        # intensity mean + quantiles + darkness mean/areas + channel ratio + coords
        base_dim = 2 + 2 * len(self.quantiles) + 2
        base_dim += 2 * len(self.dark_thresholds) + 1 + 3
        self.base_feature_dim = base_dim
        self.feature_dim = base_dim * 2

    def _cell_mean(self, values):
        pooled = F.adaptive_avg_pool2d(
            values, (self.latitude_bins, self.longitude_bins)
        )
        return pooled.flatten(2).transpose(1, 2)

    def _cell_quantiles(self, values):
        sample_height = self.latitude_bins * 4
        sample_width = self.longitude_bins * 4
        sampled = F.adaptive_avg_pool2d(values, (sample_height, sample_width))
        batch, channels = sampled.shape[:2]
        sampled = sampled.reshape(
            batch,
            channels,
            self.latitude_bins,
            4,
            self.longitude_bins,
            4,
        )
        sampled = sampled.permute(0, 2, 4, 1, 3, 5).reshape(
            batch, self.cell_count, channels, 16
        )
        ordered = sampled.float().sort(dim=-1).values
        indices = torch.as_tensor(
            [round((ordered.size(-1) - 1) * value) for value in self.quantiles],
            device=ordered.device,
            dtype=torch.long,
        )
        selected = ordered.index_select(-1, indices)
        return selected.flatten(2)

    def forward(self, images):
        if images.ndim != 5 or images.shape[2] != 2:
            raise ValueError(f"expected (B,T,2,H,W) images, got {images.shape}")
        batch, steps = images.shape[:2]
        projected, coordinates = self.reprojection(images)
        projected = projected.float()
        coordinates = coordinates.float()

        mu = coordinates[:, 2:3].clamp(0.0, 1.0)
        reference = (projected * mu).sum(dim=(-2, -1), keepdim=True)
        reference = reference / mu.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        darkness = F.relu(reference - projected) / reference.clamp_min(1.0 / 255.0)
        darkness = (darkness * torch.sqrt(mu)).clamp(0.0, 1.0)

        intensity_mean = self._cell_mean(projected)
        intensity_quantiles = self._cell_quantiles(projected)
        darkness_mean = self._cell_mean(darkness)
        dark_areas = torch.cat(
            [
                self._cell_mean((darkness >= threshold).float())
                for threshold in self.dark_thresholds
            ],
            dim=-1,
        )
        log_ratio = torch.log(
            (projected[:, 0:1] + 1.0 / 255.0)
            / (projected[:, 1:2] + 1.0 / 255.0)
        ).clamp(-4.0, 4.0) / 4.0
        ratio_mean = self._cell_mean(log_ratio)
        coordinate_mean = self._cell_mean(coordinates)

        base_features = torch.cat(
            [
                intensity_mean,
                intensity_quantiles,
                darkness_mean,
                dark_areas,
                ratio_mean,
                coordinate_mean,
            ],
            dim=-1,
        )
        base_features = base_features.reshape(batch, steps, self.cell_count, -1)
        temporal_change = torch.zeros_like(base_features)
        temporal_change[:, 1:] = base_features[:, 1:] - base_features[:, :-1]
        return torch.cat([base_features, temporal_change], dim=-1)


class CausalWindBlock(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.padding = 2 * dilation
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        residual = features
        features = F.pad(features, (self.padding, 0))
        features = self.depthwise(features)
        features = self.pointwise(features)
        features = self.dropout(F.gelu(self.norm(features)))
        return residual + features


class CausalWindEncoder(nn.Module):
    def __init__(self, wind_dim, dropout):
        super().__init__()
        self.input_projection = nn.Conv1d(4, wind_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            CausalWindBlock(wind_dim, dilation=1, dropout=dropout),
            CausalWindBlock(wind_dim, dilation=2, dropout=dropout),
            CausalWindBlock(wind_dim, dilation=4, dropout=dropout),
            CausalWindBlock(wind_dim, dilation=8, dropout=dropout),
        )
        self.output_norm = nn.LayerNorm(wind_dim)

    def forward(self, wind):
        features = make_wind_features(wind).transpose(1, 2)
        features = self.blocks(self.input_projection(features))
        return self.output_norm(features.transpose(1, 2))


class SolarWindPhysicsTransformerV5(nn.Module):
    """Causal wind forecast plus CEA coronal-hole Transformer correction."""

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        apply_solar_disk_mask=SOLAR_DISK_MASK,
        solar_disk_center_fraction=SOLAR_DISK_CENTER_FRACTION,
        solar_disk_radius_fraction=SOLAR_DISK_RADIUS_FRACTION,
        solar_cea_radius_fraction=SOLAR_CEA_RADIUS_FRACTION,
        latitude_bins=4,
        longitude_bins=8,
        d_model=96,
        wind_dim=24,
        nhead=8,
        encoder_layers=1,
        ff_dim=192,
        dropout=0.25,
        use_images=True,
        baseline_slope=None,
        baseline_intercept=None,
    ):
        super().__init__()
        if not apply_solar_disk_mask:
            raise ValueError("the CEA model requires solar disk masking")
        if solar_disk_center_fraction != (0.5, 0.5):
            raise ValueError("the CEA model requires a centered solar disk")
        if d_model <= wind_dim or d_model % nhead != 0:
            raise ValueError("invalid d_model, wind_dim, or nhead combination")

        self.use_images = use_images
        self.latitude_bins = latitude_bins
        self.longitude_bins = longitude_bins
        self.cell_count = latitude_bins * longitude_bins
        image_dim = d_model - wind_dim
        self.image_features = CoronalHoleFeatureExtractor(
            image_size,
            solar_disk_radius_fraction,
            solar_cea_radius_fraction,
            latitude_bins=latitude_bins,
            longitude_bins=longitude_bins,
        )
        self.image_projection = nn.Sequential(
            nn.Linear(self.image_features.feature_dim, image_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(image_dim * 2, image_dim),
            nn.LayerNorm(image_dim),
        )
        self.wind_encoder = CausalWindEncoder(wind_dim, dropout * 0.5)
        self.wind_residual_head = nn.Sequential(
            nn.Linear(wind_dim, wind_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(wind_dim * 2, FORECAST_STEPS),
        )

        self.memory_norm = nn.LayerNorm(d_model)
        self.observed_position = nn.Parameter(
            torch.randn(1, OBSERVED_STEPS, 1, 1, d_model) * 0.08
        )
        self.latitude_position = nn.Parameter(
            torch.randn(1, 1, latitude_bins, 1, d_model) * 0.04
        )
        self.longitude_position = nn.Parameter(
            torch.randn(1, 1, 1, longitude_bins, d_model) * 0.04
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
        self.longitude_time_encoder = nn.TransformerEncoder(
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
        self.fusion_residual_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        nn.init.normal_(self.wind_residual_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.wind_residual_head[-1].bias)
        nn.init.normal_(self.fusion_residual_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.fusion_residual_head[-1].bias)

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

    def encode_memory(self, images, wind_tokens):
        batch = images.size(0)
        if self.use_images:
            image_features = self.image_features(images)
            image_tokens = self.image_projection(image_features)
        else:
            image_tokens = wind_tokens.new_zeros(
                batch,
                OBSERVED_STEPS,
                self.cell_count,
                self.image_projection[-1].normalized_shape[0],
            )

        repeated_wind = wind_tokens.unsqueeze(2).expand(-1, -1, self.cell_count, -1)
        memory = torch.cat([image_tokens, repeated_wind], dim=-1)
        memory = memory.reshape(
            batch,
            OBSERVED_STEPS,
            self.latitude_bins,
            self.longitude_bins,
            -1,
        )
        memory = self.memory_norm(
            memory
            + self.observed_position
            + self.latitude_position
            + self.longitude_position
        )

        axial = memory.permute(0, 2, 1, 3, 4).reshape(
            batch * self.latitude_bins,
            OBSERVED_STEPS * self.longitude_bins,
            -1,
        )
        axial = self.longitude_time_encoder(axial)
        memory = axial.reshape(
            batch,
            self.latitude_bins,
            OBSERVED_STEPS,
            self.longitude_bins,
            -1,
        )
        return memory.permute(0, 2, 1, 3, 4).reshape(
            batch, OBSERVED_STEPS * self.cell_count, -1
        )

    def forward(self, images, wind, return_components=False):
        wind_tokens = self.wind_encoder(wind)
        wind_residual = self.wind_residual_head(wind_tokens[:, -1])
        wind_prediction = self.linear_baseline(wind) + wind_residual

        memory = self.encode_memory(images, wind_tokens)
        queries = self.future_queries + self.future_time_encoding
        queries = queries.expand(images.size(0), -1, -1)
        attended, _ = self.query_attention(
            queries, memory, memory, need_weights=False
        )
        queries = self.query_norm(queries + self.dropout(attended))
        queries = self.query_ffn_norm(
            queries + self.dropout(self.query_ffn(queries))
        )
        fusion_residual = self.fusion_residual_head(queries).squeeze(-1)
        prediction = wind_prediction + fusion_residual

        if return_components:
            return prediction, wind_prediction, wind_residual, fusion_residual
        return prediction
