import torch
from torch import nn

from config import IMAGE_SIZE
from model_baseline_v2_3 import SolarWindARNeuralTransformerV23
from model_solar_probabilistic import FORECAST_STEPS, OBSERVED_STEPS


class LagConstrainedFactorizedBlock(nn.Module):
    """Spatial attention plus short-range temporal attention."""

    def __init__(
        self,
        d_model,
        nhead,
        ff_dim,
        dropout,
        temporal_radius_steps,
    ):
        super().__init__()
        if temporal_radius_steps < 0:
            raise ValueError("temporal radius must be nonnegative")
        self.spatial_norm = nn.LayerNorm(d_model)
        self.temporal_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.spatial_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.temporal_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

        positions = torch.arange(OBSERVED_STEPS)
        local_mask = (
            positions[:, None] - positions[None, :]
        ).abs() > temporal_radius_steps
        self.register_buffer(
            "local_temporal_mask", local_mask, persistent=False
        )

    def _attend(self, sequence, attention, norm, attention_mask=None):
        normalized = norm(sequence)
        attended, _ = attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            need_weights=False,
        )
        return sequence + self.dropout(attended)

    def forward(self, memory):
        batch, steps, height, width, d_model = memory.shape
        cells = height * width

        spatial = memory.reshape(batch * steps, cells, d_model)
        spatial = self._attend(
            spatial, self.spatial_attention, self.spatial_norm
        )
        memory = spatial.reshape(batch, steps, height, width, d_model)

        temporal = memory.reshape(batch, steps, cells, d_model).permute(
            0, 2, 1, 3
        ).reshape(batch * cells, steps, d_model)
        temporal = self._attend(
            temporal,
            self.temporal_attention,
            self.temporal_norm,
            self.local_temporal_mask,
        )
        memory = temporal.reshape(batch, cells, steps, d_model).permute(
            0, 2, 1, 3
        ).reshape(batch, steps, height, width, d_model)

        normalized = self.ffn_norm(memory)
        return memory + self.dropout(self.ffn(normalized))


class SolarWindFixedLagMagnitudeTransformerV24(
    SolarWindARNeuralTransformerV23
):
    """AR-neural forecast with fixed-lag attentive image scaling."""

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        fixed_lag_hours=96.0,
        fixed_lag_sigma_hours=12.0,
        fixed_lag_window_hours=24.0,
        local_temporal_radius_hours=12.0,
        central_spatial_prior_strength=0.5,
        image_scale_limit=0.15,
        **kwargs,
    ):
        super().__init__(image_size=image_size, **kwargs)
        if not 72.0 <= fixed_lag_hours <= 120.0:
            raise ValueError("fixed lag must keep every source time observable")
        if fixed_lag_sigma_hours <= 0.0:
            raise ValueError("fixed lag sigma must be positive")
        if fixed_lag_window_hours < fixed_lag_sigma_hours:
            raise ValueError("fixed lag window must be at least one sigma")
        if local_temporal_radius_hours < 0.0:
            raise ValueError("local temporal radius must be nonnegative")
        if central_spatial_prior_strength < 0.0:
            raise ValueError("central spatial prior must be nonnegative")
        if not 0.0 < image_scale_limit < 1.0:
            raise ValueError("image scale limit must be between zero and one")

        self.fixed_lag_hours = float(fixed_lag_hours)
        self.fixed_lag_sigma_hours = float(fixed_lag_sigma_hours)
        self.fixed_lag_window_hours = float(fixed_lag_window_hours)
        self.local_temporal_radius_hours = float(
            local_temporal_radius_hours
        )
        self.central_spatial_prior_strength = float(
            central_spatial_prior_strength
        )
        self.image_scale_limit = float(image_scale_limit)
        self.timing_prior_raw.requires_grad_(False)

        d_model = int(kwargs.get("d_model", 128))
        nhead = int(kwargs.get("nhead", 4))
        ff_dim = int(kwargs.get("ff_dim", 256))
        dropout = float(kwargs.get("dropout", 0.20))
        encoder_layers = int(kwargs.get("encoder_layers", 1))
        temporal_radius_steps = int(
            round(local_temporal_radius_hours / 6.0)
        )
        self.factorized_blocks = nn.ModuleList(
            [
                LagConstrainedFactorizedBlock(
                    d_model,
                    nhead,
                    ff_dim,
                    dropout,
                    temporal_radius_steps,
                )
                for _ in range(encoder_layers)
            ]
        )

        rows = torch.linspace(-1.0, 1.0, self.spatial_height)
        columns = torch.linspace(-1.0, 1.0, self.spatial_width)
        spatial_distance = rows[:, None].square() + columns[None, :].square()
        self.register_buffer(
            "central_spatial_log_prior",
            -self.central_spatial_prior_strength * spatial_distance,
            persistent=False,
        )

        del self.image_residual_head
        self.image_scale_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.image_scale_head.weight)
        nn.init.zeros_(self.image_scale_head.bias)

    def _timing_bias(self, wind, dtype):
        source_age = (
            self.fixed_lag_hours - self.horizon_hours
        ).to(dtype=dtype)
        age_error = (
            self.image_age_hours[None, :].to(dtype=dtype)
            - source_age[:, None]
        )
        temporal_bias = -0.5 * (
            age_error / self.fixed_lag_sigma_hours
        ).square()
        outside_window = age_error.abs() > self.fixed_lag_window_hours
        temporal_bias = temporal_bias.masked_fill(
            outside_window, torch.finfo(dtype).min
        )
        combined_bias = (
            temporal_bias[:, :, None, None]
            + self.central_spatial_log_prior.to(dtype=dtype)[None, None]
        ).reshape(FORECAST_STEPS, -1)
        batch_bias = combined_bias.unsqueeze(0).expand(
            wind.shape[0], -1, -1
        )
        attention_bias = batch_bias.repeat_interleave(self.nhead, dim=0)
        fixed_transit = wind.new_full(
            (wind.shape[0],), self.fixed_lag_hours
        )
        fixed_strength = wind.new_tensor(1.0)
        return attention_bias, fixed_transit, fixed_strength

    def forward(
        self,
        images,
        wind,
        return_components=False,
        return_diagnostics=False,
    ):
        if not self.use_images:
            return super().forward(
                images,
                wind,
                return_components=return_components,
                return_diagnostics=return_diagnostics,
            )
        if wind.ndim != 2 or wind.shape[1] != OBSERVED_STEPS:
            raise ValueError(
                f"expected wind shaped (B,{OBSERVED_STEPS}), "
                f"got {tuple(wind.shape)}"
            )

        wind_features = self.wind_encoder(wind)
        wind_residual = self.wind_residual_head(wind_features)
        wind_prediction = self.linear_baseline(wind) + wind_residual

        memory, image_keep, masked, delta = self.encode_images(images)
        queries = self.future_queries + self.future_time_encoding
        queries = queries.expand(images.shape[0], -1, -1)
        queries = queries + self.wind_query_projection(wind_features).unsqueeze(1)
        timing_bias, fixed_transit, prior_strength = self._timing_bias(
            wind, queries.dtype
        )
        decoded, attention_weights = self.forecast_decoder(
            queries,
            memory,
            memory_mask=timing_bias,
            need_weights=True,
        )
        attention = self._attention_statistics(attention_weights)

        repeated_wind = wind_features.unsqueeze(1).expand(
            -1, FORECAST_STEPS, -1
        )
        fused = self.fusion(torch.cat([decoded, repeated_wind], dim=-1))
        raw_scale = self.image_scale_head(fused).squeeze(-1)
        scale_fraction = self.image_scale_limit * torch.tanh(raw_scale)
        scale_residual = wind_prediction.detach() * scale_fraction
        residual_cap = (
            self.baseline_residual_scale.to(scale_residual.dtype)
            * self.residual_cap_multiplier
        )
        bounded_residual = residual_cap * torch.tanh(
            scale_residual / residual_cap.clamp_min(1e-6)
        )
        image_gate = torch.sigmoid(self.image_gate_head(fused).squeeze(-1))
        fusion_residual = (
            bounded_residual * image_gate * image_keep.squeeze(-1)
        )
        prediction = wind_prediction + fusion_residual

        self._last_training_diagnostics = {
            "attention_age_h": attention["expected_age"].detach().mean(),
            "attention_entropy": attention["temporal_entropy"].detach().mean(),
            "spatial_attention_entropy": attention[
                "spatial_entropy"
            ].detach().mean(),
            "fixed_lag_h": fixed_transit.detach().mean(),
            "image_gate": image_gate.detach().mean(),
            "image_scale_percent": (
                scale_fraction.detach().abs().mean() * 100.0
            ),
            "delta_rms": delta.detach().float().square().mean().sqrt(),
        }

        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "attention_weights": attention["temporal_weights"],
                "attention_token_weights": attention["token_weights"],
                "attention_spatial_weights": attention["spatial_weights"],
                "attention_expected_age_hours": attention["expected_age"],
                "attention_entropy": attention["temporal_entropy"],
                "attention_spatial_entropy": attention["spatial_entropy"],
                "nominal_transit_hours": fixed_transit,
                "image_gate": image_gate,
                "image_scale_fraction": scale_fraction,
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
