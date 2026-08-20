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
from model import Inception3D, make_solar_disk_mask, spatial_block_average
from model_solar_probabilistic import FORECAST_STEPS, OBSERVED_STEPS
from model_tile_transformer import sinusoidal_time_encoding


AU_TRAVEL_HOURS_AT_1000_KM_S = 149_597_870.7 / 1000.0 / 3600.0


def inverse_softplus(value):
    if value <= 0.0:
        return -20.0
    return math.log(math.expm1(value))


class BaselineImageEncoder(nn.Module):
    """Official Inception3D front end with masked intensity and frame deltas."""

    def __init__(
        self,
        image_size,
        center_fraction,
        radius_fraction,
        spatial_height,
        spatial_width,
        delta_gain,
    ):
        super().__init__()
        if spatial_height <= 0 or spatial_width <= 0:
            raise ValueError("spatial dimensions must be positive")
        if delta_gain <= 0.0:
            raise ValueError("delta_gain must be positive")

        self.center_fraction = tuple(center_fraction)
        self.radius_fraction = float(radius_fraction)
        self.spatial_height = int(spatial_height)
        self.spatial_width = int(spatial_width)
        self.delta_gain = float(delta_gain)

        mask = make_solar_disk_mask(
            image_size,
            image_size,
            self.center_fraction,
            self.radius_fraction,
        )
        self.register_buffer("solar_disk_mask", mask, persistent=False)

        # The official baseline starts with two intensity channels. V2.1 adds
        # signed differences for the same 193A/211A channels.
        self.stem = nn.Sequential(
            nn.Conv3d(4, 32, (1, 5, 5), padding=(0, 2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(
                (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
            ),
        )
        blocks = []
        in_channels = 32
        for _ in range(3):
            blocks.extend(
                [
                    Inception3D(in_channels, 32),
                    nn.MaxPool3d(
                        (1, 3, 3),
                        stride=(1, 2, 2),
                        padding=(0, 1, 1),
                    ),
                ]
            )
            in_channels = 128
        self.image_encoder = nn.Sequential(*blocks)

    def _mask(self, images):
        if self.solar_disk_mask.shape[-2:] == images.shape[-2:]:
            mask = self.solar_disk_mask.to(
                device=images.device, dtype=images.dtype
            )
        else:
            mask = make_solar_disk_mask(
                images.shape[-2],
                images.shape[-1],
                self.center_fraction,
                self.radius_fraction,
                device=images.device,
                dtype=images.dtype,
            )
        return images * mask

    def preprocess(self, images):
        masked = self._mask(images)
        delta = torch.zeros_like(masked)
        delta[:, 1:] = masked[:, 1:] - masked[:, :-1]
        delta = torch.clamp(delta * self.delta_gain, -1.0, 1.0)
        return masked, delta

    def forward(self, images):
        if images.ndim != 5 or images.shape[1:3] != (OBSERVED_STEPS, 2):
            raise ValueError(
                "expected images shaped "
                f"(B,{OBSERVED_STEPS},2,H,W), got {tuple(images.shape)}"
            )
        masked, delta = self.preprocess(images)
        features = torch.cat([masked, delta], dim=2)
        features = features.permute(0, 2, 1, 3, 4).contiguous()
        features = self.image_encoder(self.stem(features))
        if self.spatial_height != self.spatial_width:
            raise ValueError("V2.1 currently requires a square spatial grid")
        features = spatial_block_average(features, self.spatial_height)
        features = features.permute(0, 2, 1, 3, 4).flatten(2)
        return features, masked, delta


class ForecastCrossAttentionBlock(nn.Module):
    """One compact decoder block with inspectable cross-attention."""

    def __init__(self, d_model, nhead, ff_dim, dropout):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.self_norm = nn.LayerNorm(d_model)
        self.cross_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, memory, memory_mask=None, need_weights=False):
        attended, _ = self.self_attention(
            queries, queries, queries, need_weights=False
        )
        queries = self.self_norm(queries + self.dropout(attended))
        attended, weights = self.cross_attention(
            queries,
            memory,
            memory,
            attn_mask=memory_mask,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        queries = self.cross_norm(queries + self.dropout(attended))
        queries = self.ffn_norm(queries + self.dropout(self.ffn(queries)))
        return queries, weights


class SolarWindBaselineTransformerV21(nn.Module):
    """Baseline CNN plus delta images and a regularized Transformer back end."""

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        apply_solar_disk_mask=SOLAR_DISK_MASK,
        solar_disk_center_fraction=SOLAR_DISK_CENTER_FRACTION,
        solar_disk_radius_fraction=SOLAR_DISK_RADIUS_FRACTION,
        solar_cea_radius_fraction=None,
        latitude_bins=4,
        longitude_bins=4,
        d_model=96,
        wind_dim=64,
        nhead=4,
        encoder_layers=1,
        ff_dim=192,
        dropout=0.20,
        use_images=True,
        baseline_slope=None,
        baseline_intercept=None,
        baseline_residual_scale=None,
        delta_gain=4.0,
        image_time_mask_probability=0.15,
        image_modality_drop_probability=0.25,
        timing_prior_strength=0.10,
        timing_prior_sigma_hours=36.0,
        residual_cap_multiplier=1.5,
    ):
        super().__init__()
        del solar_cea_radius_fraction
        if not apply_solar_disk_mask:
            raise ValueError("V2.1 requires solar disk masking")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if encoder_layers <= 0:
            raise ValueError("encoder_layers must be positive")
        if wind_dim <= 0 or ff_dim <= 0:
            raise ValueError("wind_dim and ff_dim must be positive")
        if not 0.0 <= image_time_mask_probability < 1.0:
            raise ValueError("image_time_mask_probability must be in [0, 1)")
        if not 0.0 <= image_modality_drop_probability < 1.0:
            raise ValueError(
                "image_modality_drop_probability must be in [0, 1)"
            )
        if timing_prior_sigma_hours <= 0.0:
            raise ValueError("timing_prior_sigma_hours must be positive")
        if residual_cap_multiplier <= 0.0:
            raise ValueError("residual_cap_multiplier must be positive")

        self.use_images = bool(use_images)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.image_time_mask_probability = float(
            image_time_mask_probability
        )
        self.image_modality_drop_probability = float(
            image_modality_drop_probability
        )
        self.timing_prior_sigma_hours = float(timing_prior_sigma_hours)
        self.residual_cap_multiplier = float(residual_cap_multiplier)

        self.image_encoder = BaselineImageEncoder(
            image_size=image_size,
            center_fraction=solar_disk_center_fraction,
            radius_fraction=solar_disk_radius_fraction,
            spatial_height=latitude_bins,
            spatial_width=longitude_bins,
            delta_gain=delta_gain,
        )
        image_feature_dim = 128 * latitude_bins * longitude_bins
        self.image_projection = nn.Sequential(
            nn.Linear(image_feature_dim, d_model),
            nn.LayerNorm(d_model),
        )
        observed_positions = torch.arange(-OBSERVED_STEPS + 1, 1)
        self.register_buffer(
            "observed_time_encoding",
            sinusoidal_time_encoding(observed_positions, d_model),
            persistent=False,
        )
        self.position_scale = nn.Parameter(torch.tensor(0.25))
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

        # This is the official baseline's whole-history wind MLP, kept separate
        # from image timestamps because L1 wind and EUV pixels are not causally
        # aligned at the same row index.
        self.wind_encoder = nn.Sequential(
            nn.Linear(OBSERVED_STEPS, 128),
            nn.SELU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, wind_dim),
            nn.SELU(inplace=True),
        )
        self.wind_residual_head = nn.Sequential(
            nn.Linear(wind_dim, 64),
            nn.SELU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, FORECAST_STEPS),
        )
        self.wind_query_projection = nn.Sequential(
            nn.Linear(wind_dim, d_model),
            nn.LayerNorm(d_model),
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
        self.forecast_decoder = ForecastCrossAttentionBlock(
            d_model, nhead, ff_dim, dropout
        )

        image_age_hours = torch.arange(
            OBSERVED_STEPS - 1, -1, -1, dtype=torch.float32
        ) * 6.0
        horizon_hours = torch.arange(
            1, FORECAST_STEPS + 1, dtype=torch.float32
        ) * 6.0
        self.register_buffer(
            "image_age_hours", image_age_hours, persistent=False
        )
        self.register_buffer(
            "horizon_hours", horizon_hours, persistent=False
        )
        self.timing_prior_raw = nn.Parameter(
            torch.tensor(inverse_softplus(float(timing_prior_strength)))
        )

        fusion_dim = d_model + wind_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.GELU(),
        )
        self.image_residual_head = nn.Linear(d_model, 1)
        self.image_gate_head = nn.Linear(d_model, 1)

        if baseline_slope is None:
            baseline_slope = torch.ones(FORECAST_STEPS)
        if baseline_intercept is None:
            baseline_intercept = torch.zeros(FORECAST_STEPS)
        if baseline_residual_scale is None:
            baseline_residual_scale = [0.1] * FORECAST_STEPS
        if len(baseline_slope) != FORECAST_STEPS:
            raise ValueError("baseline_slope must have 12 values")
        if len(baseline_intercept) != FORECAST_STEPS:
            raise ValueError("baseline_intercept must have 12 values")
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
            torch.as_tensor(
                baseline_residual_scale, dtype=torch.float32
            ).view(1, -1),
        )

        nn.init.zeros_(self.wind_residual_head[-1].weight)
        nn.init.zeros_(self.wind_residual_head[-1].bias)
        nn.init.normal_(self.image_residual_head.weight, std=1e-3)
        nn.init.zeros_(self.image_residual_head.bias)
        nn.init.zeros_(self.image_gate_head.weight)
        nn.init.constant_(self.image_gate_head.bias, -1.5)
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

    def _timing_bias(self, wind, dtype):
        # wind is scaled by 1/1000 in the dataset. This converts the latest
        # observed speed to a soft 1-AU travel-time prior, never a hard match.
        nominal_transit = AU_TRAVEL_HOURS_AT_1000_KM_S / wind[:, -1]
        nominal_transit = nominal_transit.clamp(48.0, 144.0)
        expected_age = (
            nominal_transit[:, None] - self.horizon_hours[None, :]
        ).clamp(0.0, (OBSERVED_STEPS - 1) * 6.0)
        error = (
            self.image_age_hours[None, None, :]
            - expected_age[:, :, None]
        )
        strength = F.softplus(self.timing_prior_raw)
        bias = -0.5 * strength * (
            error / self.timing_prior_sigma_hours
        ).square()
        return (
            bias.to(dtype=dtype).repeat_interleave(self.nhead, dim=0),
            nominal_transit,
            strength,
        )

    def encode_images(self, images):
        features, masked, delta = self.image_encoder(images)
        tokens = self.image_projection(features)
        if self.training and self.image_time_mask_probability > 0.0:
            time_keep = (
                torch.rand(
                    tokens.shape[0], OBSERVED_STEPS, 1, device=tokens.device
                )
                >= self.image_time_mask_probability
            ).to(dtype=tokens.dtype)
            tokens = tokens * time_keep
        keep = self._image_keep_mask(
            tokens.shape[0], tokens.device, tokens.dtype
        )
        tokens = tokens * keep
        tokens = tokens + self.position_scale * self.observed_time_encoding
        return self.temporal_encoder(tokens), keep, masked, delta

    def training_diagnostics(self):
        return self._last_training_diagnostics

    def forward(
        self,
        images,
        wind,
        return_components=False,
        return_diagnostics=False,
    ):
        if wind.ndim != 2 or wind.shape[1] != OBSERVED_STEPS:
            raise ValueError(
                f"expected wind shaped (B,{OBSERVED_STEPS}), got {tuple(wind.shape)}"
            )

        wind_features = self.wind_encoder(wind)
        wind_residual = self.wind_residual_head(wind_features)
        wind_prediction = self.linear_baseline(wind) + wind_residual

        if not self.use_images:
            fusion_residual = torch.zeros_like(wind_prediction)
            prediction = wind_prediction
            self._last_training_diagnostics = {
                "attention_age_h": wind.new_tensor(0.0),
                "attention_entropy": wind.new_tensor(0.0),
                "nominal_transit_h": wind.new_tensor(0.0),
                "image_gate": wind.new_tensor(0.0),
                "timing_prior_strength": F.softplus(
                    self.timing_prior_raw.detach()
                ),
                "delta_rms": wind.new_tensor(0.0),
            }
            components = (
                prediction,
                wind_prediction,
                wind_residual,
                fusion_residual,
            )
            if return_components and return_diagnostics:
                return (*components, None)
            if return_components:
                return components
            if return_diagnostics:
                return prediction, None
            return prediction

        memory, image_keep, masked, delta = self.encode_images(images)
        queries = self.future_queries + self.future_time_encoding
        queries = queries.expand(images.shape[0], -1, -1)
        queries = queries + self.wind_query_projection(wind_features).unsqueeze(1)
        timing_bias, nominal_transit, prior_strength = self._timing_bias(
            wind, queries.dtype
        )
        decoded, attention_weights = self.forecast_decoder(
            queries,
            memory,
            memory_mask=timing_bias,
            need_weights=True,
        )
        diagnostic_attention = attention_weights / attention_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        repeated_wind = wind_features.unsqueeze(1).expand(
            -1, FORECAST_STEPS, -1
        )
        fused = self.fusion(torch.cat([decoded, repeated_wind], dim=-1))
        raw_residual = self.image_residual_head(fused).squeeze(-1)
        image_gate = torch.sigmoid(self.image_gate_head(fused).squeeze(-1))
        residual_cap = (
            self.baseline_residual_scale.to(raw_residual.dtype)
            * self.residual_cap_multiplier
        )
        bounded_residual = residual_cap * torch.tanh(
            raw_residual / residual_cap.clamp_min(1e-6)
        )
        fusion_residual = (
            bounded_residual * image_gate * image_keep.squeeze(-1)
        )
        prediction = wind_prediction + fusion_residual

        entropy = -torch.sum(
            diagnostic_attention
            * torch.log(diagnostic_attention.clamp_min(1e-8)),
            dim=-1,
        ) / math.log(OBSERVED_STEPS)
        expected_age = torch.sum(
            diagnostic_attention
            * self.image_age_hours.to(diagnostic_attention.dtype).view(1, 1, -1),
            dim=-1,
        )
        self._last_training_diagnostics = {
            "attention_age_h": expected_age.detach().mean(),
            "attention_entropy": entropy.detach().mean(),
            "nominal_transit_h": nominal_transit.detach().mean(),
            "image_gate": image_gate.detach().mean(),
            "timing_prior_strength": prior_strength.detach(),
            "delta_rms": delta.detach().float().square().mean().sqrt(),
        }

        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "attention_weights": diagnostic_attention,
                "attention_expected_age_hours": expected_age,
                "attention_entropy": entropy,
                "nominal_transit_hours": nominal_transit,
                "image_gate": image_gate,
                "timing_prior_strength": prior_strength,
                "masked_images": masked,
                "delta_images": delta,
                "image_memory": memory,
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
