import math

import torch
from torch import nn

from model_solar_deformable_timing_v14 import (
    FORECAST_STEPS,
    HINDCAST_STEPS,
    SolarWindDeformableTimingV14,
)


ARCHITECTURE_NAME = "SolarWindPeakEventV15"
FILE_STEM = "solar_peak_event_v15"


def _logit(probability):
    probability = min(max(float(probability), 1e-4), 1.0 - 1e-4)
    return math.log(probability / (1.0 - probability))


class SolarWindPeakEventV15(SolarWindDeformableTimingV14):
    """V14 with explicitly supervised future peak-time and peak-value heads."""

    def __init__(
        self,
        peak_hidden_dim=96,
        peak_curve_sigma_steps=1.25,
        peak_value_min=0.25,
        peak_value_max=0.90,
        maximum_peak_blend=0.30,
        initial_peak_blend=0.05,
        peak_correction_cap_multiplier=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if peak_hidden_dim <= 0:
            raise ValueError("peak_hidden_dim must be positive")
        if peak_curve_sigma_steps <= 0.0:
            raise ValueError("peak_curve_sigma_steps must be positive")
        if not 0.0 < peak_value_min < peak_value_max:
            raise ValueError("invalid peak value interval")
        if not 0.0 < initial_peak_blend <= maximum_peak_blend <= 1.0:
            raise ValueError("peak blend must satisfy 0 < initial <= max <= 1")
        if peak_correction_cap_multiplier <= 0.0:
            raise ValueError("peak correction cap multiplier must be positive")

        self.peak_hidden_dim = int(peak_hidden_dim)
        self.peak_curve_sigma_steps = float(peak_curve_sigma_steps)
        self.peak_value_min = float(peak_value_min)
        self.peak_value_max = float(peak_value_max)
        self.maximum_peak_blend = float(maximum_peak_blend)
        self.peak_correction_cap_multiplier = float(
            peak_correction_cap_multiplier
        )

        self.peak_scalar_projection = nn.Linear(4, self.d_model, bias=False)
        self.peak_context = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.peak_hidden_dim),
            nn.GELU(),
            nn.Dropout(kwargs.get("dropout", 0.15)),
        )
        self.peak_time_head = nn.Linear(self.peak_hidden_dim, 1)
        self.peak_value_head = nn.Sequential(
            nn.Linear(self.peak_hidden_dim, self.peak_hidden_dim),
            nn.GELU(),
            nn.Linear(self.peak_hidden_dim, 1),
        )
        self.peak_blend_raw = nn.Parameter(
            torch.tensor(
                _logit(initial_peak_blend / self.maximum_peak_blend),
                dtype=torch.float32,
            )
        )

        nn.init.zeros_(self.peak_time_head.weight)
        nn.init.zeros_(self.peak_time_head.bias)
        nn.init.zeros_(self.peak_value_head[-1].weight)
        nn.init.zeros_(self.peak_value_head[-1].bias)
        horizon_index = torch.arange(FORECAST_STEPS, dtype=torch.float32)
        squared_distance = (
            horizon_index.view(FORECAST_STEPS, 1)
            - horizon_index.view(1, FORECAST_STEPS)
        ).square()
        peak_curve_bank = torch.exp(
            -squared_distance / (2.0 * self.peak_curve_sigma_steps**2)
        )
        self.register_buffer("peak_curve_bank", peak_curve_bank)

    def peak_blend_strength(self):
        return self.maximum_peak_blend * torch.sigmoid(self.peak_blend_raw)

    def forward(
        self,
        images,
        wind,
        return_components=False,
        return_aux=False,
        time_keep=None,
        image_keep=None,
    ):
        v14_prediction, components, aux = super().forward(
            images,
            wind,
            return_components=True,
            return_aux=True,
            time_keep=time_keep,
            image_keep=image_keep,
        )
        future_query = aux["query_features"][:, HINDCAST_STEPS:]
        source_future = components["source_future"]
        ar_base = components["ar_base"]
        horizon_fraction = self.horizon_hours.to(
            device=wind.device, dtype=wind.dtype
        ) / self.horizon_hours[-1].to(device=wind.device, dtype=wind.dtype)
        scalar_features = torch.stack(
            [
                ar_base,
                source_future,
                source_future - ar_base,
                horizon_fraction.unsqueeze(0).expand_as(ar_base),
            ],
            dim=-1,
        )
        peak_features = self.peak_context(
            future_query + self.peak_scalar_projection(scalar_features)
        )
        peak_time_logits = self.peak_time_head(peak_features).squeeze(-1)
        peak_time_probability = torch.softmax(peak_time_logits, dim=-1)
        pooled_peak_features = torch.einsum(
            "bh,bhd->bd", peak_time_probability, peak_features
        )
        peak_value_fraction = torch.sigmoid(
            self.peak_value_head(pooled_peak_features).squeeze(-1)
        )
        peak_value = self.peak_value_min + (
            self.peak_value_max - self.peak_value_min
        ) * peak_value_fraction

        event_curve = peak_time_probability @ self.peak_curve_bank.to(
            dtype=wind.dtype
        )
        base_at_peak = (peak_time_probability * v14_prediction).sum(dim=-1)
        peak_cap = self.peak_correction_cap_multiplier * (
            peak_time_probability
            * self.baseline_residual_scale.to(dtype=wind.dtype).unsqueeze(0)
        ).sum(dim=-1)
        bounded_peak_delta = peak_cap * torch.tanh(
            (peak_value - base_at_peak) / peak_cap.clamp_min(1e-4)
        )
        peak_event_gate = (
            self.peak_blend_strength().to(dtype=wind.dtype)
            * aux["image_keep"]
        )
        peak_event_correction = (
            peak_event_gate.unsqueeze(-1)
            * event_curve
            * bounded_peak_delta.unsqueeze(-1)
        )
        prediction = v14_prediction + peak_event_correction
        expected_peak_hour = (
            peak_time_probability
            * self.horizon_hours.to(device=wind.device, dtype=wind.dtype)
        ).sum(dim=-1)

        v14_image_correction = components["image_correction"]
        total_image_correction = prediction - ar_base
        components.update(
            {
                "v14_prediction": v14_prediction,
                "v14_image_correction": v14_image_correction,
                "peak_event_correction": peak_event_correction,
                "peak_value": peak_value,
                "image_correction": total_image_correction,
            }
        )
        aux.update(
            {
                "v14_prediction": v14_prediction,
                "peak_features": peak_features,
                "peak_time_logits": peak_time_logits,
                "peak_time_probability": peak_time_probability,
                "peak_value": peak_value,
                "expected_peak_hour": expected_peak_hour,
                "peak_event_curve": event_curve,
                "peak_event_gate": peak_event_gate,
            }
        )
        self._last_diagnostics.update(
            {
                "predicted_peak_hour": expected_peak_hour.mean(),
                "peak_time_entropy": (
                    -(
                        peak_time_probability.clamp_min(1e-8).log()
                        * peak_time_probability
                    ).sum(dim=-1)
                    / math.log(FORECAST_STEPS)
                ).mean(),
                "predicted_peak_value_kms": peak_value.mean() * 1000.0,
                "peak_event_gate": peak_event_gate.mean(),
                "peak_event_correction_rms_kms": torch.sqrt(
                    peak_event_correction.square().mean()
                )
                * 1000.0,
                "image_correction_rms_kms": torch.sqrt(
                    total_image_correction.square().mean()
                )
                * 1000.0,
            }
        )
        if return_components and return_aux:
            return prediction, components, aux
        if return_components:
            return prediction, components
        if return_aux:
            return prediction, aux
        return prediction
