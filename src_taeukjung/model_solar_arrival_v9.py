import math

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
from model_solar_geometry_v3 import SolarGeometryEncoderV3
from model_solar_probabilistic import FORECAST_STEPS, OBSERVED_STEPS
from model_tile_transformer import make_wind_features, sinusoidal_time_encoding


class CausalTemporalBlock(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.left_padding = 2 * dilation
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        residual = features
        features = features.transpose(1, 2)
        features = F.pad(features, (self.left_padding, 0))
        features = self.pointwise(self.depthwise(features)).transpose(1, 2)
        features = self.dropout(F.gelu(features))
        return self.norm(residual + features)


class CausalTemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, dropout):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                CausalTemporalBlock(hidden_dim, 2**index, dropout)
                for index in range(layers)
            ]
        )

    def forward(self, features):
        features = self.input_projection(features)
        for block in self.blocks:
            features = block(features)
        return features


class SolarWindArrivalTCNV9(nn.Module):
    """Separate image/wind TCNs with a learned per-image arrival-time gate."""

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        apply_solar_disk_mask=SOLAR_DISK_MASK,
        solar_disk_center_fraction=SOLAR_DISK_CENTER_FRACTION,
        solar_disk_radius_fraction=SOLAR_DISK_RADIUS_FRACTION,
        solar_cea_radius_fraction=SOLAR_CEA_RADIUS_FRACTION,
        latitude_bins=2,
        longitude_bins=4,
        d_model=96,
        wind_dim=24,
        nhead=8,
        encoder_layers=1,
        ff_dim=192,
        dropout=0.15,
        use_images=True,
        baseline_slope=None,
        baseline_intercept=None,
        baseline_residual_scale=None,
        image_cnn_channels=48,
        temporal_layers=3,
        visual_dropout=0.10,
        image_time_mask_probability=0.10,
        image_modality_drop_probability=0.10,
        transit_min_hours=48.0,
        transit_max_hours=120.0,
        arrival_sigma_hours=24.0,
        arrival_prior_strength=0.25,
        residual_cap_multiplier=2.5,
    ):
        super().__init__()
        del nhead
        if not apply_solar_disk_mask:
            raise ValueError("the CEA model requires solar disk masking")
        if solar_disk_center_fraction != (0.5, 0.5):
            raise ValueError("the CEA model requires a centered solar disk")
        if latitude_bins <= 0 or longitude_bins <= 0:
            raise ValueError("latitude_bins and longitude_bins must be positive")
        if d_model % 2 != 0 or wind_dim <= 0:
            raise ValueError("d_model must be even and wind_dim must be positive")
        if temporal_layers <= 0:
            raise ValueError("temporal_layers must be positive")
        if not 0.0 <= image_time_mask_probability < 1.0:
            raise ValueError("image_time_mask_probability must be in [0, 1)")
        if not 0.0 <= image_modality_drop_probability < 1.0:
            raise ValueError("image_modality_drop_probability must be in [0, 1)")
        if not 0.0 < transit_min_hours < transit_max_hours:
            raise ValueError("invalid transit-hour range")
        if arrival_sigma_hours <= 0.0 or arrival_prior_strength < 0.0:
            raise ValueError("invalid arrival prior configuration")
        if residual_cap_multiplier <= 0.0:
            raise ValueError("residual_cap_multiplier must be positive")

        self.use_images = bool(use_images)
        self.latitude_bins = int(latitude_bins)
        self.longitude_bins = int(longitude_bins)
        self.cell_count = self.latitude_bins * self.longitude_bins
        self.d_model = int(d_model)
        self.wind_hidden_dim = max(int(wind_dim) * 2, 32)
        self.image_time_mask_probability = float(image_time_mask_probability)
        self.image_modality_drop_probability = float(
            image_modality_drop_probability
        )
        self.transit_min_hours = float(transit_min_hours)
        self.transit_max_hours = float(transit_max_hours)
        self.residual_cap_multiplier = float(residual_cap_multiplier)

        self.image_encoder = SolarGeometryEncoderV3(
            image_size,
            solar_disk_radius_fraction,
            solar_cea_radius_fraction,
            image_cnn_channels,
            spatial_height=self.latitude_bins,
            spatial_width=self.longitude_bins,
            visual_dropout=visual_dropout,
        )
        self.image_temporal_encoder = CausalTemporalEncoder(
            self.cell_count * image_cnn_channels,
            d_model,
            temporal_layers,
            dropout,
        )
        self.wind_temporal_encoder = CausalTemporalEncoder(
            4,
            self.wind_hidden_dim,
            temporal_layers,
            dropout * 0.5,
        )

        self.wind_residual_head = nn.Sequential(
            nn.Linear(self.wind_hidden_dim, ff_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim // 2, FORECAST_STEPS),
        )

        attention_dim = d_model // 2
        self.image_key = nn.Linear(d_model, attention_dim)
        self.future_query = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, attention_dim),
        )
        self.transit_head = nn.Linear(d_model, 1)
        self.source_score_head = nn.Linear(d_model, 1)

        future_positions = torch.arange(1, FORECAST_STEPS + 1)
        self.register_buffer(
            "future_time_encoding",
            sinusoidal_time_encoding(future_positions, d_model),
            persistent=False,
        )
        self.future_embedding = nn.Parameter(
            torch.randn(1, FORECAST_STEPS, d_model) * 0.02
        )
        observed_age_hours = torch.arange(
            OBSERVED_STEPS - 1, -1, -1, dtype=torch.float32
        ) * 6.0
        horizon_hours = torch.arange(
            1, FORECAST_STEPS + 1, dtype=torch.float32
        ) * 6.0
        self.register_buffer(
            "observed_age_hours", observed_age_hours, persistent=False
        )
        self.register_buffer("horizon_hours", horizon_hours, persistent=False)

        self.log_arrival_sigma = nn.Parameter(
            torch.log(torch.tensor(float(arrival_sigma_hours)))
        )
        prior_raw = (
            math.log(math.expm1(float(arrival_prior_strength)))
            if arrival_prior_strength > 0.0
            else -20.0
        )
        self.arrival_prior_raw = nn.Parameter(torch.tensor(prior_raw))

        fusion_input_dim = d_model * 2 + self.wind_hidden_dim
        self.fusion_norm = nn.LayerNorm(fusion_input_dim)
        self.fusion_trunk = nn.Sequential(
            nn.Linear(fusion_input_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.GELU(),
        )
        self.fusion_residual_head = nn.Linear(d_model, 1)
        self.fusion_gate_head = nn.Linear(d_model, 1)

        if baseline_slope is None:
            baseline_slope = torch.ones(FORECAST_STEPS)
        if baseline_intercept is None:
            baseline_intercept = torch.zeros(FORECAST_STEPS)
        if baseline_residual_scale is None:
            baseline_residual_scale = [0.1] * FORECAST_STEPS
        if len(baseline_residual_scale) != FORECAST_STEPS:
            raise ValueError("baseline_residual_scale must have 12 values")
        self.register_buffer(
            "baseline_slope",
            torch.as_tensor(baseline_slope, dtype=torch.float32).view(1, -1),
        )
        self.register_buffer(
            "baseline_intercept",
            torch.as_tensor(baseline_intercept, dtype=torch.float32).view(1, -1),
        )
        self.register_buffer(
            "baseline_residual_scale",
            torch.as_tensor(baseline_residual_scale, dtype=torch.float32).view(
                1, -1
            ),
        )

        nn.init.zeros_(self.wind_residual_head[-1].weight)
        nn.init.zeros_(self.wind_residual_head[-1].bias)
        nn.init.zeros_(self.transit_head.weight)
        nn.init.zeros_(self.transit_head.bias)
        nn.init.zeros_(self.source_score_head.weight)
        nn.init.zeros_(self.source_score_head.bias)
        nn.init.normal_(self.fusion_residual_head.weight, std=1e-3)
        nn.init.zeros_(self.fusion_residual_head.bias)
        nn.init.zeros_(self.fusion_gate_head.weight)
        nn.init.constant_(self.fusion_gate_head.bias, -1.0)

        self._last_training_diagnostics = {}

    def linear_baseline(self, wind):
        return wind[:, -1:] * self.baseline_slope + self.baseline_intercept

    def _image_keep_mask(self, batch, device, dtype):
        if not self.training or self.image_modality_drop_probability <= 0.0:
            return torch.ones(batch, 1, 1, device=device, dtype=dtype)
        return (
            torch.rand(batch, 1, 1, device=device)
            >= self.image_modality_drop_probability
        ).to(dtype=dtype)

    def encode_images(self, images):
        batch = images.size(0)
        if not self.use_images:
            zeros = images.new_zeros(batch, OBSERVED_STEPS, self.d_model)
            keep = images.new_zeros(batch, 1, 1)
            return zeros, keep

        spatial = self.image_encoder(images).flatten(2)
        if self.training and self.image_time_mask_probability > 0.0:
            time_keep = (
                torch.rand(batch, OBSERVED_STEPS, 1, device=images.device)
                >= self.image_time_mask_probability
            ).to(dtype=spatial.dtype)
            spatial = spatial * time_keep

        keep = self._image_keep_mask(batch, images.device, spatial.dtype)
        temporal = self.image_temporal_encoder(spatial) * keep
        return temporal, keep

    def arrival_aggregation(self, image_tokens):
        keys = self.image_key(image_tokens)
        future = self.future_embedding + self.future_time_encoding
        queries = self.future_query(future).expand(image_tokens.size(0), -1, -1)
        content_logits = torch.einsum("bhd,btd->bht", queries, keys)
        content_logits = content_logits / math.sqrt(keys.size(-1))

        transit_range = self.transit_max_hours - self.transit_min_hours
        transit_hours = self.transit_min_hours + transit_range * torch.sigmoid(
            self.transit_head(image_tokens).squeeze(-1)
        )
        source_scores = self.source_score_head(image_tokens).squeeze(-1)

        dtype = content_logits.dtype
        observed_age = self.observed_age_hours.to(dtype=dtype)
        horizon = self.horizon_hours.to(dtype=dtype)
        arrival_hours = transit_hours.to(dtype=dtype) - observed_age.view(1, -1)
        timing_error = arrival_hours.unsqueeze(1) - horizon.view(1, -1, 1)
        sigma = self.log_arrival_sigma.exp().clamp(6.0, 72.0).to(dtype=dtype)
        prior_strength = F.softplus(self.arrival_prior_raw).to(dtype=dtype)
        timing_bias = -0.5 * prior_strength * (timing_error / sigma).square()

        logits = content_logits.float()
        logits = logits + source_scores.unsqueeze(1).float() + timing_bias.float()
        weights = torch.softmax(logits, dim=-1)
        context = torch.einsum(
            "bht,btd->bhd", weights.to(dtype=image_tokens.dtype), image_tokens
        )
        return {
            "context": context,
            "weights": weights,
            "transit_hours": transit_hours.float(),
            "source_probability": torch.sigmoid(source_scores.float()),
            "arrival_hours": arrival_hours.float(),
            "prior_strength": prior_strength.float(),
            "sigma_hours": sigma.float(),
        }

    def training_diagnostics(self):
        return self._last_training_diagnostics

    def forward(
        self,
        images,
        wind,
        return_components=False,
        return_diagnostics=False,
    ):
        wind_tokens = self.wind_temporal_encoder(make_wind_features(wind))
        wind_state = wind_tokens[:, -1]
        wind_residual = self.wind_residual_head(wind_state)
        wind_prediction = self.linear_baseline(wind) + wind_residual

        image_tokens, image_keep = self.encode_images(images)
        arrival = self.arrival_aggregation(image_tokens)
        future = (self.future_embedding + self.future_time_encoding).expand(
            images.size(0), -1, -1
        )
        repeated_wind = wind_state.unsqueeze(1).expand(-1, FORECAST_STEPS, -1)
        fused = torch.cat([arrival["context"], repeated_wind, future], dim=-1)
        fused = self.fusion_trunk(self.fusion_norm(fused))

        raw_residual = self.fusion_residual_head(fused).squeeze(-1)
        fusion_gate = torch.sigmoid(self.fusion_gate_head(fused).squeeze(-1))
        residual_cap = self.baseline_residual_scale.to(raw_residual.dtype)
        residual_cap = residual_cap * self.residual_cap_multiplier
        bounded_residual = residual_cap * torch.tanh(raw_residual / residual_cap)
        fusion_residual = bounded_residual * fusion_gate * image_keep.squeeze(-1)
        prediction = wind_prediction + fusion_residual

        entropy = -torch.sum(
            arrival["weights"] * torch.log(arrival["weights"].clamp_min(1e-8)),
            dim=-1,
        ) / math.log(OBSERVED_STEPS)
        expected_age = torch.sum(
            arrival["weights"]
            * self.observed_age_hours.float().view(1, 1, -1),
            dim=-1,
        )
        self._last_training_diagnostics = {
            "arrival_age_h": expected_age.detach().mean(),
            "arrival_entropy": entropy.detach().mean(),
            "transit_h": arrival["transit_hours"].detach().mean(),
            "source_probability": arrival["source_probability"].detach().mean(),
            "fusion_gate": fusion_gate.detach().mean(),
            "arrival_prior_strength": arrival["prior_strength"].detach(),
            "arrival_sigma_h": arrival["sigma_hours"].detach(),
        }

        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "arrival_weights": arrival["weights"],
                "transit_hours": arrival["transit_hours"],
                "source_probability": arrival["source_probability"],
                "arrival_hours": arrival["arrival_hours"],
                "expected_image_age_hours": expected_age,
                "arrival_entropy": entropy,
                "fusion_gate": fusion_gate,
                "image_tokens": image_tokens,
                "arrival_prior_strength": arrival["prior_strength"],
                "arrival_sigma_hours": arrival["sigma_hours"],
            }

        components = (
            prediction,
            wind_prediction,
            wind_residual,
            fusion_residual,
        )
        if return_components and return_diagnostics:
            return (*components, diagnostics)
        if return_components:
            return components
        if return_diagnostics:
            return prediction, diagnostics
        return prediction
