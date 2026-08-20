import math

import torch
from torch.nn import functional as F

from model_solar_cnn_v6 import SolarWindCNNTransformerV6
from model_solar_probabilistic import FORECAST_STEPS, OBSERVED_STEPS


class SolarWindBallisticTransformerV8(SolarWindCNNTransformerV6):
    """V6 with a speed-conditioned solar-rotation prior and bounded correction."""

    def __init__(
        self,
        *args,
        baseline_residual_scale=None,
        physics_prior_strength=1.0,
        longitude_sigma_degrees=30.0,
        latitude_sigma_degrees=45.0,
        residual_cap_multiplier=2.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if baseline_residual_scale is None:
            baseline_residual_scale = [0.1] * FORECAST_STEPS
        if len(baseline_residual_scale) != FORECAST_STEPS:
            raise ValueError("baseline_residual_scale must have 12 values")
        if physics_prior_strength < 0.0:
            raise ValueError("physics_prior_strength must be non-negative")
        if longitude_sigma_degrees <= 0.0 or latitude_sigma_degrees <= 0.0:
            raise ValueError("physics-prior angular scales must be positive")
        if residual_cap_multiplier <= 0.0:
            raise ValueError("residual_cap_multiplier must be positive")

        self.attention_heads = kwargs.get("nhead", 8)
        self.longitude_sigma_degrees = float(longitude_sigma_degrees)
        self.latitude_sigma_degrees = float(latitude_sigma_degrees)
        self.residual_cap_multiplier = float(residual_cap_multiplier)

        prior_raw = (
            math.log(math.expm1(float(physics_prior_strength)))
            if physics_prior_strength > 0.0
            else -20.0
        )
        self.physics_prior_raw = torch.nn.Parameter(torch.tensor(prior_raw))
        self.register_buffer(
            "baseline_residual_scale",
            torch.as_tensor(baseline_residual_scale, dtype=torch.float32).view(1, -1),
        )

        observed_age_hours = torch.arange(
            OBSERVED_STEPS - 1, -1, -1, dtype=torch.float32
        ) * 6.0
        future_hours = torch.arange(1, FORECAST_STEPS + 1, dtype=torch.float32) * 6.0
        longitude = (
            torch.arange(self.longitude_bins, dtype=torch.float32) + 0.5
        ) * (180.0 / self.longitude_bins) - 90.0
        sin_latitude = (
            torch.arange(self.latitude_bins, dtype=torch.float32) + 0.5
        ) * (2.0 / self.latitude_bins) - 1.0
        latitude = torch.rad2deg(torch.asin(sin_latitude))
        self.register_buffer("observed_age_hours", observed_age_hours, persistent=False)
        self.register_buffer("future_hours", future_hours, persistent=False)
        self.register_buffer("cell_longitude_degrees", longitude, persistent=False)
        self.register_buffer("cell_latitude_degrees", latitude, persistent=False)

    def physics_attention_bias(self, wind_prediction, dtype):
        # The ballistic transit estimate is a routing prior, not a second target path.
        speed_km_s = (wind_prediction.detach().float() * 1000.0).clamp(250.0, 800.0)
        transit_hours = 149_597_870.7 / (speed_km_s * 3600.0)

        delta_hours = (
            self.future_hours.view(1, FORECAST_STEPS, 1)
            - transit_hours.unsqueeze(-1)
            + self.observed_age_hours.view(1, 1, OBSERVED_STEPS)
        )
        synodic_rotation_degrees_per_hour = 360.0 / (27.2753 * 24.0)
        expected_longitude = -synodic_rotation_degrees_per_hour * delta_hours

        longitude_error = (
            self.cell_longitude_degrees.view(1, 1, 1, 1, self.longitude_bins)
            - expected_longitude.unsqueeze(-1).unsqueeze(-1)
        )
        latitude_error = self.cell_latitude_degrees.view(
            1, 1, 1, self.latitude_bins, 1
        )
        log_prior = -0.5 * (
            longitude_error / self.longitude_sigma_degrees
        ).square() - 0.5 * (latitude_error / self.latitude_sigma_degrees).square()
        strength = F.softplus(self.physics_prior_raw.float())
        spatial_bias = strength * log_prior
        spatial_bias = spatial_bias.expand(
            -1, -1, -1, -1, self.longitude_bins
        ).reshape(
            wind_prediction.size(0),
            FORECAST_STEPS,
            OBSERVED_STEPS * self.cell_count,
        )
        temporal_bias = (strength * log_prior).amax(dim=(-2, -1))
        bias = torch.cat([temporal_bias, spatial_bias], dim=-1)
        bias = bias.repeat_interleave(self.attention_heads, dim=0)
        return bias.to(dtype=dtype)

    def encode_memory(self, images, wind_tokens):
        batch = images.size(0)
        if self.use_images:
            image_tokens = self.image_encoder(images)
        else:
            image_tokens = wind_tokens.new_zeros(
                batch,
                wind_tokens.size(1),
                self.cell_count,
                self.image_token_dim,
            )

        repeated_wind = wind_tokens.unsqueeze(2).expand(-1, -1, self.cell_count, -1)
        memory = torch.cat([image_tokens, repeated_wind], dim=-1).reshape(
            batch,
            image_tokens.size(1),
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

        temporal_memory = self.longitude_time_encoder(memory.mean(dim=(2, 3)))
        spatial_memory = memory.reshape(
            batch, wind_tokens.size(1) * self.cell_count, -1
        )
        return torch.cat([temporal_memory, spatial_memory], dim=1)

    def forward(self, images, wind, return_components=False):
        wind_tokens = self.wind_encoder(wind)
        wind_residual = self.wind_residual_head(wind_tokens[:, -1])
        wind_prediction = self.linear_baseline(wind) + wind_residual

        memory = self.encode_memory(images, wind_tokens)
        queries = self.future_queries + self.future_time_encoding
        queries = queries.expand(images.size(0), -1, -1)
        attention_bias = self.physics_attention_bias(wind_prediction, queries.dtype)
        attended, _ = self.query_attention(
            queries,
            memory,
            memory,
            attn_mask=attention_bias,
            need_weights=False,
        )
        queries = self.query_norm(queries + self.dropout(attended))
        queries = self.query_ffn_norm(
            queries + self.dropout(self.query_ffn(queries))
        )

        raw_residual = self.fusion_residual_head(queries).squeeze(-1)
        residual_cap = self.baseline_residual_scale.to(raw_residual.dtype)
        residual_cap = residual_cap * self.residual_cap_multiplier
        fusion_residual = residual_cap * torch.tanh(raw_residual / residual_cap)
        prediction = wind_prediction + fusion_residual

        if return_components:
            return prediction, wind_prediction, wind_residual, fusion_residual
        return prediction
