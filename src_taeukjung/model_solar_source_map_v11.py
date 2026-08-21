import math

import torch
from torch import nn
from torch.nn import functional as F

from src_torazang65.model import Inception3D, conv_block


FORECAST_STEPS = 12
OBSERVED_STEPS = 20
ARCHITECTURE_NAME = "SolarWindSourceMapV11"
FILE_STEM = "solar_source_map_v11"


class MPSCompatibleSpatialPool(nn.Module):
    """Pool a CNN grid to 2x4 using operations supported by MPS."""

    def forward(self, features):
        height, width = features.shape[-2:]
        if height % 2 != 0 or width % 4 != 0:
            raise ValueError(
                f"CNN grid {height}x{width} cannot be evenly pooled to 2x4"
            )
        kernel = (1, height // 2, width // 4)
        return F.avg_pool3d(features, kernel_size=kernel, stride=kernel)


def make_soft_solar_disk_mask(
    image_size,
    center_fraction=(0.5, 0.5),
    radius_fraction=0.49,
    edge_pixels=1.5,
):
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if not 0.0 < radius_fraction <= 0.5:
        raise ValueError("radius_fraction must be in (0, 0.5]")
    if edge_pixels < 0.0:
        raise ValueError("edge_pixels must be nonnegative")

    center_y = (image_size - 1) * float(center_fraction[0])
    center_x = (image_size - 1) * float(center_fraction[1])
    radius = image_size * float(radius_fraction)
    y = torch.arange(image_size, dtype=torch.float32).view(image_size, 1)
    x = torch.arange(image_size, dtype=torch.float32).view(1, image_size)
    distance = torch.sqrt((x - center_x).square() + (y - center_y).square())
    if edge_pixels == 0.0:
        mask = (distance <= radius).float()
    else:
        phase = ((radius - distance) / float(edge_pixels) + 1.0).clamp(0.0, 1.0)
        mask = 0.5 - 0.5 * torch.cos(math.pi * phase)
    return mask.view(1, 1, 1, image_size, image_size)


class SolarWindSourceMapV11(nn.Module):
    """AR(2)-anchored, cell-resolved ballistic solar-wind forecaster."""

    def __init__(
        self,
        image_size=64,
        use_images=True,
        ar_coefficients=None,
        ar_intercept=0.0,
        ar_ridge_strength=30.0,
        baseline_residual_scale=None,
        source_hidden_dim=64,
        dropout=0.10,
        time_mask_prob=0.05,
        modality_drop_prob=0.15,
        propagation_cap_multiplier=1.25,
        fixed_lag_hours=96.0,
        fixed_lag_reference_speed_kms=430.0,
        delta_gain=4.0,
        apply_solar_disk_mask=True,
        solar_disk_center_fraction=(0.5, 0.5),
        solar_disk_radius_fraction=0.49,
        solar_disk_edge_pixels=1.5,
        climatology_speed_kms=430.0,
        kernel_sigma_hours=12.0,
        fallback_weight=1.0,
        transit_residual_hours=18.0,
        fast_wind_threshold_kms=550.0,
        fast_wind_scale_kms=50.0,
        fast_quiet_suppression=0.50,
        source_head_init_std=1e-3,
    ):
        super().__init__()
        if ar_coefficients is None:
            ar_coefficients = [0.0, 1.0]
        if baseline_residual_scale is None:
            baseline_residual_scale = [0.08] * FORECAST_STEPS
        coefficients = torch.as_tensor(ar_coefficients, dtype=torch.float32)
        residual_scale = torch.as_tensor(
            baseline_residual_scale, dtype=torch.float32
        )
        if coefficients.ndim != 1 or not 1 <= len(coefficients) <= OBSERVED_STEPS:
            raise ValueError("invalid AR coefficients")
        if residual_scale.shape != (FORECAST_STEPS,) or torch.any(
            residual_scale <= 0
        ):
            raise ValueError("baseline_residual_scale must contain 12 positives")
        if source_hidden_dim <= 0:
            raise ValueError("source_hidden_dim must be positive")
        if not 0.0 <= time_mask_prob < 1.0:
            raise ValueError("time_mask_prob must be in [0, 1)")
        if not 0.0 <= modality_drop_prob < 1.0:
            raise ValueError("modality_drop_prob must be in [0, 1)")
        if propagation_cap_multiplier <= 0.0:
            raise ValueError("propagation_cap_multiplier must be positive")
        if not 72.0 <= fixed_lag_hours <= 120.0:
            raise ValueError("fixed_lag_hours must be between 72 and 120")
        if not 250.0 <= fixed_lag_reference_speed_kms <= 900.0:
            raise ValueError("fixed_lag_reference_speed_kms is invalid")
        if delta_gain <= 0.0 or kernel_sigma_hours <= 0.0:
            raise ValueError("delta_gain and kernel_sigma_hours must be positive")
        if fallback_weight <= 0.0 or transit_residual_hours < 0.0:
            raise ValueError("invalid fallback or transit residual setting")
        if fast_wind_scale_kms <= 0.0:
            raise ValueError("fast_wind_scale_kms must be positive")
        if not 0.0 <= fast_quiet_suppression <= 1.0:
            raise ValueError("fast_quiet_suppression must be in [0, 1]")

        self.image_size = int(image_size)
        self.use_images = bool(use_images)
        self.ar_order = int(len(coefficients))
        self.ar_ridge_strength = float(ar_ridge_strength)
        self.time_mask_prob = float(time_mask_prob)
        self.modality_drop_prob = float(modality_drop_prob)
        self.propagation_cap_multiplier = float(propagation_cap_multiplier)
        self.fixed_lag_hours = float(fixed_lag_hours)
        self.fixed_lag_reference_speed_kms = float(
            fixed_lag_reference_speed_kms
        )
        self.delta_gain = float(delta_gain)
        self.apply_solar_disk_mask = bool(apply_solar_disk_mask)
        self.kernel_sigma_hours = float(kernel_sigma_hours)
        self.fallback_weight = float(fallback_weight)
        self.transit_residual_hours = float(transit_residual_hours)
        self.fast_wind_threshold = float(fast_wind_threshold_kms) / 1000.0
        self.fast_wind_scale = float(fast_wind_scale_kms) / 1000.0
        self.fast_quiet_suppression = float(fast_quiet_suppression)

        self.register_buffer("ar_coefficients", coefficients)
        self.register_buffer(
            "ar_intercept", torch.tensor(float(ar_intercept), dtype=torch.float32)
        )
        self.register_buffer("baseline_residual_scale", residual_scale.view(1, -1))
        self.register_buffer(
            "climatology",
            torch.tensor(float(climatology_speed_kms) / 1000.0),
        )
        self.register_buffer(
            "dist_eff_hours_speed",
            torch.tensor(
                float(fixed_lag_hours)
                * float(fixed_lag_reference_speed_kms)
                / 1000.0
            ),
        )
        self.register_buffer(
            "solar_disk_mask",
            make_soft_solar_disk_mask(
                image_size,
                center_fraction=solar_disk_center_fraction,
                radius_fraction=solar_disk_radius_fraction,
                edge_pixels=solar_disk_edge_pixels,
            ),
        )
        self.register_buffer(
            "image_age_hours",
            torch.arange(OBSERVED_STEPS - 1, -1, -1, dtype=torch.float32) * 6.0,
        )
        self.register_buffer(
            "hindcast_hours", torch.arange(-12, 1, dtype=torch.float32) * 6.0
        )
        self.register_buffer(
            "horizon_hours",
            torch.arange(1, FORECAST_STEPS + 1, dtype=torch.float32) * 6.0,
        )
        self.register_buffer(
            "cell_lon_deg", torch.tensor([-67.5, -22.5, 22.5, 67.5])
        )
        self.register_buffer("cell_lat_norm", torch.tensor([0.5, -0.5]))
        self.omega_deg_per_hour = 360.0 / (27.2753 * 24.0)

        self.stem = nn.Sequential(
            conv_block(4, 32, (1, 5, 5), padding=(0, 2, 2)),
            nn.MaxPool3d(
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
        )
        blocks = []
        in_channels = 32
        for _ in range(3):
            blocks.extend(
                [
                    Inception3D(in_channels, 32),
                    nn.MaxPool3d(
                        kernel_size=(1, 3, 3),
                        stride=(1, 2, 2),
                        padding=(0, 1, 1),
                    ),
                ]
            )
            in_channels = 128
        self.image_encoder = nn.Sequential(*blocks)
        self.spatial_pool = MPSCompatibleSpatialPool()

        self.source_context = nn.Sequential(
            nn.LayerNorm(130),
            nn.Linear(130, source_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.source_speed_head = nn.Linear(source_hidden_dim, 1)
        self.source_gate_head = nn.Linear(source_hidden_dim, 1)
        self.transit_residual_head = nn.Linear(source_hidden_dim, 1)
        self.lon_offset_head = nn.Linear(source_hidden_dim, 1)

        self.image_summary_projection = nn.Linear(256, 32, bias=False)
        self.surge_head = nn.Sequential(
            nn.Linear(256, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.fusion_gate_head = nn.Linear(32 + 4 + 1, FORECAST_STEPS)

        for head in (
            self.source_speed_head,
            self.source_gate_head,
            self.transit_residual_head,
            self.lon_offset_head,
        ):
            nn.init.normal_(head.weight, std=float(source_head_init_std))
            nn.init.zeros_(head.bias)
        nn.init.constant_(self.source_speed_head.bias, -0.96)
        nn.init.zeros_(self.fusion_gate_head.weight)
        nn.init.constant_(self.fusion_gate_head.bias, -2.0)

        self._last_diagnostics = {}

    def _recursive_ar(self, wind):
        history = [wind[:, index] for index in range(wind.shape[1])]
        predictions = []
        coefficients = self.ar_coefficients.to(dtype=wind.dtype)
        intercept = self.ar_intercept.to(dtype=wind.dtype)
        for _ in range(FORECAST_STEPS):
            context = torch.stack(history[-self.ar_order :], dim=1)
            next_value = intercept + (context * coefficients).sum(dim=1)
            history.append(next_value)
            predictions.append(next_value)
        return torch.stack(predictions, dim=1)

    def _encode_cells(self, images):
        if self.apply_solar_disk_mask:
            images = images * self.solar_disk_mask.to(dtype=images.dtype)
        difference = torch.zeros_like(images)
        difference[:, 1:] = images[:, 1:] - images[:, :-1]
        image_channels = torch.cat([images, self.delta_gain * difference], dim=2)
        features = image_channels.permute(0, 2, 1, 3, 4).contiguous()
        features = self.image_encoder(self.stem(features))
        features = self.spatial_pool(features)
        return features.permute(0, 2, 3, 4, 1).contiguous()

    def _source_map(self, cell_features, wind):
        batch_size = cell_features.shape[0]
        time_keep = torch.ones(
            batch_size, OBSERVED_STEPS, 1, 1, 1,
            device=cell_features.device,
            dtype=cell_features.dtype,
        )
        if self.training and self.time_mask_prob > 0.0:
            time_keep = (
                torch.rand(
                    batch_size,
                    OBSERVED_STEPS,
                    1,
                    1,
                    1,
                    device=cell_features.device,
                )
                >= self.time_mask_prob
            ).to(dtype=cell_features.dtype)

        image_keep = torch.ones(
            batch_size, device=cell_features.device, dtype=cell_features.dtype
        )
        if self.training and self.modality_drop_prob > 0.0:
            image_keep = (
                torch.rand(batch_size, device=cell_features.device)
                >= self.modality_drop_prob
            ).to(dtype=cell_features.dtype)
        keep = time_keep * image_keep.view(batch_size, 1, 1, 1, 1)
        cell_features = cell_features * keep

        lat = self.cell_lat_norm.view(1, 1, 2, 1, 1).expand(
            batch_size, OBSERVED_STEPS, 2, 4, 1
        )
        lon = (self.cell_lon_deg / 90.0).view(1, 1, 1, 4, 1).expand(
            batch_size, OBSERVED_STEPS, 2, 4, 1
        )
        context = self.source_context(torch.cat([cell_features, lat, lon], dim=-1))
        source_speed = 0.25 + 0.65 * torch.sigmoid(
            self.source_speed_head(context)
        ).squeeze(-1)
        source_gate = F.softplus(self.source_gate_head(context)).squeeze(-1)
        source_gate = source_gate * keep.squeeze(-1)
        transit_residual = torch.tanh(
            self.transit_residual_head(context)
        ).squeeze(-1)
        source_lon = self.cell_lon_deg.view(1, 1, 1, 4) + 22.5 * torch.tanh(
            self.lon_offset_head(context)
        ).squeeze(-1)

        rotation_wait = -source_lon / self.omega_deg_per_hour
        transit = (
            self.dist_eff_hours_speed.to(dtype=source_speed.dtype) / source_speed
            + self.transit_residual_hours * transit_residual
        )
        arrival = (
            rotation_wait
            + transit
            - self.image_age_hours.view(1, OBSERVED_STEPS, 1, 1)
        )
        time_grid = torch.cat([self.hindcast_hours, self.horizon_hours])
        kernel = torch.exp(
            -(
                time_grid.view(1, 1, 1, 1, -1) - arrival.unsqueeze(-1)
            ).square()
            / (2.0 * self.kernel_sigma_hours**2)
        )
        source_weight = source_gate.unsqueeze(-1) * kernel
        weight_sum = source_weight.sum(dim=(1, 2, 3))
        fallback = torch.as_tensor(
            self.fallback_weight, device=wind.device, dtype=wind.dtype
        )
        source_prediction = (
            (
                source_weight * source_speed.unsqueeze(-1)
            ).sum(dim=(1, 2, 3))
            + fallback * self.climatology.to(dtype=wind.dtype)
        ) / (weight_sum + fallback)
        coverage = weight_sum / (weight_sum + fallback)

        frame_summary = cell_features.mean(dim=(2, 3))
        image_summary = torch.cat(
            [
                frame_summary.mean(dim=1),
                frame_summary[:, -5:].mean(dim=1)
                - frame_summary[:, :5].mean(dim=1),
            ],
            dim=1,
        )
        surge_logit = self.surge_head(image_summary)
        surge_probability = torch.sigmoid(surge_logit)
        image_embedding = F.gelu(self.image_summary_projection(image_summary))
        wind_summary = torch.stack(
            [
                wind[:, -1],
                wind.mean(dim=1),
                wind.std(dim=1),
                wind[:, -1] - wind[:, 0],
            ],
            dim=1,
        )
        fusion_alpha = torch.sigmoid(
            self.fusion_gate_head(
                torch.cat([image_embedding, wind_summary, surge_probability], dim=1)
            )
        )
        fast_probability = torch.sigmoid(
            (wind[:, -1:] - self.fast_wind_threshold) / self.fast_wind_scale
        )
        quiet_fast_factor = 1.0 - self.fast_quiet_suppression * fast_probability * (
            1.0 - surge_probability
        )
        fusion_alpha = (
            fusion_alpha
            * coverage[:, -FORECAST_STEPS:]
            * quiet_fast_factor
            * image_keep.unsqueeze(-1)
        )
        return {
            "source_prediction": source_prediction,
            "hindcast": source_prediction[:, :13],
            "future": source_prediction[:, 13:],
            "fusion_alpha": fusion_alpha,
            "image_keep": image_keep,
            "transit_residual": transit_residual,
            "source_weight": source_weight,
            "source_speed": source_speed,
            "source_gate": source_gate,
            "source_lon": source_lon,
            "arrival": arrival,
            "coverage": coverage,
            "surge_logit": surge_logit,
            "surge_probability": surge_probability,
        }

    def forward(
        self,
        images,
        wind,
        return_components=False,
        return_aux=False,
    ):
        ar_baseline = self._recursive_ar(wind)
        batch_size = wind.shape[0]
        if self.use_images:
            source = self._source_map(self._encode_cells(images), wind)
            raw_propagation = source["fusion_alpha"] * (
                source["future"] - ar_baseline
            )
            cap = (
                self.baseline_residual_scale.to(dtype=raw_propagation.dtype)
                * self.propagation_cap_multiplier
            )
            propagation = cap * torch.tanh(raw_propagation / cap.clamp_min(1e-6))
            prediction = ar_baseline + propagation
            self._last_diagnostics = {
                "source_speed_mean_kms": source["source_speed"].mean() * 1000.0,
                "source_speed_std_kms": source["source_speed"].std() * 1000.0,
                "arrival_mean_h": source["arrival"].mean(),
                "arrival_std_h": source["arrival"].std(),
                "source_gate_mean": source["source_gate"].mean(),
                "coverage_hind_mean": source["coverage"][:, :13].mean(),
                "coverage_future_mean": source["coverage"][:, 13:].mean(),
                "fusion_alpha_mean": source["fusion_alpha"].mean(),
                "surge_probability_mean": source["surge_probability"].mean(),
                "source_lon_offset_rms_deg": (
                    source["source_lon"]
                    - self.cell_lon_deg.view(1, 1, 1, 4)
                ).square().mean().sqrt(),
                "propagation_cap_saturation": (
                    raw_propagation.abs() >= 0.95 * cap
                ).float().mean(),
            }
        else:
            source = None
            propagation = torch.zeros_like(ar_baseline)
            prediction = ar_baseline
            self._last_diagnostics = {}

        components = {
            "ar_baseline": ar_baseline,
            "wind_prediction": ar_baseline,
            "source_prediction": (
                source["future"]
                if source is not None
                else self.climatology.expand(batch_size, FORECAST_STEPS)
            ),
            "fusion_alpha": (
                source["fusion_alpha"]
                if source is not None
                else torch.zeros_like(ar_baseline)
            ),
            "propagation_residual": propagation,
            "correction": torch.zeros_like(ar_baseline),
            "surge_probability": (
                source["surge_probability"]
                if source is not None
                else torch.zeros(batch_size, 1, device=wind.device, dtype=wind.dtype)
            ),
        }
        aux = {
            "hindcast": source["hindcast"] if source is not None else None,
            "image_keep": source["image_keep"] if source is not None else None,
            "transit_residual": (
                source["transit_residual"] if source is not None else None
            ),
            "source_weight": source["source_weight"] if source is not None else None,
            "surge_logit": source["surge_logit"] if source is not None else None,
        }

        if return_components and return_aux:
            return prediction, components, aux
        if return_components:
            return prediction, components
        if return_aux:
            return prediction, aux
        return prediction

    def training_diagnostics(self):
        return self._last_diagnostics
