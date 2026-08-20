import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src_torazang65.model import SolarWindBaseline as SeokhoPropagationV5B


FORECAST_STEPS = 12
OBSERVED_STEPS = 20
ARCHITECTURE_NAME = "SolarWindAnchoredHybridV10"
FILE_STEM = "solar_hybrid_v10"


class OutputCapture(nn.Module):
    """Wrap a module while retaining its current differentiable output."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.latest = None

    def forward(self, *args, **kwargs):
        self.latest = self.module(*args, **kwargs)
        return self.latest


class LevelDifferenceWindEncoder(nn.Module):
    def __init__(self, output_dim, dropout):
        super().__init__()
        feature_dim = 1 + OBSERVED_STEPS + (OBSERVED_STEPS - 1)
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.SELU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, output_dim),
            nn.SELU(inplace=True),
        )

    def forward(self, wind):
        latest = wind[:, -1:]
        relative_history = wind - latest
        differences = wind[:, 1:] - wind[:, :-1]
        return self.network(
            torch.cat([latest, relative_history, differences], dim=1)
        )


class BoundedResidualHead(nn.Module):
    def __init__(self, input_dim, output_cap, dropout, init_std=2e-4):
        super().__init__()
        cap = torch.as_tensor(output_cap, dtype=torch.float32).view(1, -1)
        if cap.shape[1] != FORECAST_STEPS or torch.any(cap <= 0):
            raise ValueError("output_cap must contain 12 positive values")
        self.register_buffer("cap", cap)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SELU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, FORECAST_STEPS),
        )
        nn.init.normal_(self.network[-1].weight, std=float(init_std))
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):
        raw = self.network(features)
        cap = self.cap.to(dtype=raw.dtype)
        return cap * torch.tanh(raw / cap.clamp_min(1e-6))


class MPSCompatibleSpatialPool(nn.Module):
    """Pool to 2x4 without AdaptiveAvgPool3d, which MPS lacks."""

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


class SolarWindAnchoredHybridV10(SeokhoPropagationV5B):
    """Seokho V5b propagation with an AR baseline and anchored image residuals."""

    def __init__(
        self,
        image_size=64,
        use_images=True,
        ar_coefficients=None,
        ar_intercept=0.0,
        ar_ridge_strength=30.0,
        baseline_residual_scale=None,
        wind_feature_dim=64,
        wind_residual_cap_multiplier=1.0,
        propagation_cap_multiplier=1.25,
        correction_cap_multiplier=0.75,
        correction_drop_prob=0.30,
        fixed_lag_hours=96.0,
        fixed_lag_reference_speed_kms=430.0,
        delta_gain=4.0,
        apply_solar_disk_mask=True,
        solar_disk_center_fraction=(0.5, 0.5),
        solar_disk_radius_fraction=0.49,
        solar_disk_edge_pixels=1.5,
        climatology_speed_kms=430.0,
        source_head_init_std=1e-3,
        correction_head_init_std=2e-4,
        **kwargs,
    ):
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
        if residual_scale.shape != (FORECAST_STEPS,) or torch.any(residual_scale <= 0):
            raise ValueError("baseline_residual_scale must contain 12 positives")
        if not 72.0 <= fixed_lag_hours <= 120.0:
            raise ValueError("fixed_lag_hours must be between 72 and 120")
        if not 250.0 <= fixed_lag_reference_speed_kms <= 900.0:
            raise ValueError("fixed_lag_reference_speed_kms is outside the source range")
        if delta_gain <= 0.0:
            raise ValueError("delta_gain must be positive")
        if not 0.0 <= correction_drop_prob < 1.0:
            raise ValueError("correction_drop_prob must be in [0, 1)")

        parent_kwargs = dict(kwargs)
        parent_kwargs.pop("image_in_channels", None)
        parent_kwargs.pop("add_diff_channels", None)
        parent_kwargs.pop("correction_drop_prob", None)
        super().__init__(
            image_size=image_size,
            use_images=use_images,
            image_in_channels=4,
            add_diff_channels=False,
            correction_drop_prob=0.0,
            **parent_kwargs,
        )

        self.image_size = int(image_size)
        self.spatial_pool = MPSCompatibleSpatialPool()
        self.ar_order = int(len(coefficients))
        self.ar_ridge_strength = float(ar_ridge_strength)
        self.wind_residual_cap_multiplier = float(wind_residual_cap_multiplier)
        self.propagation_cap_multiplier = float(propagation_cap_multiplier)
        self.correction_cap_multiplier = float(correction_cap_multiplier)
        self.hybrid_correction_drop_prob = float(correction_drop_prob)
        self.fixed_lag_hours = float(fixed_lag_hours)
        self.fixed_lag_reference_speed_kms = float(
            fixed_lag_reference_speed_kms
        )
        self.delta_gain = float(delta_gain)
        self.apply_solar_disk_mask = bool(apply_solar_disk_mask)

        self.register_buffer("ar_coefficients", coefficients)
        self.register_buffer(
            "ar_intercept",
            torch.tensor(float(ar_intercept), dtype=torch.float32),
        )
        self.register_buffer(
            "hybrid_residual_scale", residual_scale.view(1, -1)
        )
        self.register_buffer(
            "solar_disk_mask",
            make_soft_solar_disk_mask(
                image_size,
                center_fraction=solar_disk_center_fraction,
                radius_fraction=solar_disk_radius_fraction,
                edge_pixels=solar_disk_edge_pixels,
            ),
            persistent=False,
        )

        dropout = float(parent_kwargs.get("dropout", 0.1))
        self.hybrid_wind_encoder = LevelDifferenceWindEncoder(
            int(wind_feature_dim), dropout
        )
        self.hybrid_wind_residual_head = BoundedResidualHead(
            int(wind_feature_dim),
            residual_scale * float(wind_residual_cap_multiplier),
            dropout,
        )

        self.output_head = OutputCapture(self.output_head)
        self.fusion_gate_head = OutputCapture(self.fusion_gate_head)

        nn.init.normal_(
            self.output_head.module[-1].weight,
            std=float(correction_head_init_std),
        )
        nn.init.zeros_(self.output_head.module[-1].bias)
        nn.init.normal_(
            self.fusion_gate_head.module.weight,
            std=float(source_head_init_std),
        )
        nn.init.constant_(self.fusion_gate_head.module.bias, -2.0)
        for head in (
            self.source_speed_head,
            self.source_gate_head,
            self.transit_residual_head,
        ):
            nn.init.normal_(head.weight, std=float(source_head_init_std))

        with torch.no_grad():
            self.climatology.fill_(float(climatology_speed_kms) / 1000.0)
            distance_coefficient = (
                self.fixed_lag_hours
                * self.fixed_lag_reference_speed_kms
                / 1000.0
            )
            distance_fraction = (distance_coefficient - 30.0) / 25.0
            distance_fraction = min(max(distance_fraction, 1e-4), 1.0 - 1e-4)
            self.dist_eff_raw.fill_(
                math.log(distance_fraction / (1.0 - distance_fraction))
            )

        # The empirical four-day lag fixes the global distance scale. Image
        # tokens still predict speed-dependent transit and a bounded residual.
        self.dist_eff_raw.requires_grad_(False)
        self.reversion_logit.requires_grad_(False)
        self._last_hybrid_diagnostics = {}

    def recursive_ar_baseline(self, wind):
        history = [wind[:, index] for index in range(wind.shape[1])]
        coefficients = self.ar_coefficients.to(dtype=wind.dtype)
        intercept = self.ar_intercept.to(dtype=wind.dtype)
        predictions = []
        for _ in range(FORECAST_STEPS):
            context = torch.stack(history[-self.ar_order :], dim=1)
            next_value = intercept + context @ coefficients
            history.append(next_value)
            predictions.append(next_value)
        return torch.stack(predictions, dim=1)

    def _prepare_images(self, images):
        if images.ndim != 5 or images.shape[1] != OBSERVED_STEPS:
            raise ValueError(
                "expected images shaped (B,20,C,H,W), "
                f"got {tuple(images.shape)}"
            )
        if images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"expected {self.image_size}px images, got {tuple(images.shape[-2:])}"
            )
        if self.apply_solar_disk_mask:
            images = images * self.solar_disk_mask.to(
                device=images.device, dtype=images.dtype
            )
        difference = torch.zeros_like(images)
        difference[:, 1:] = images[:, 1:] - images[:, :-1]
        augmented = torch.cat(
            [images, difference * self.delta_gain], dim=2
        )
        return augmented, difference

    def _bounded_component(self, value, multiplier):
        cap = self.hybrid_residual_scale.to(dtype=value.dtype) * float(multiplier)
        return cap * torch.tanh(value / cap.clamp_min(1e-6))

    def forward(
        self,
        images,
        wind,
        return_components=False,
        return_aux=False,
    ):
        if wind.ndim != 2 or wind.shape[1] != OBSERVED_STEPS:
            raise ValueError(
                f"expected wind shaped (B,{OBSERVED_STEPS}), got {tuple(wind.shape)}"
            )

        augmented_images, difference = self._prepare_images(images)
        self.output_head.latest = None
        self.fusion_gate_head.latest = None
        parent_prediction, parent_aux = super().forward(
            augmented_images, wind, return_aux=True
        )
        raw_correction = self.output_head.latest.squeeze(-1)

        ar_baseline = self.recursive_ar_baseline(wind)
        wind_features = self.hybrid_wind_encoder(wind)
        wind_residual = self.hybrid_wind_residual_head(wind_features)
        wind_prediction = ar_baseline + wind_residual

        if self.use_images:
            alpha = torch.sigmoid(self.fusion_gate_head.latest)
            beta = torch.sigmoid(self.reversion_logit)
            old_base = wind[:, -1:] + beta * (
                self.climatology - wind[:, -1:]
            )
            old_propagation = parent_prediction - old_base - raw_correction
            # Remove the v5a shrinkage term. A quiet image forecast equal to
            # climatology must contribute zero instead of pulling slow wind up.
            propagation_anomaly = old_propagation - alpha * (
                self.climatology - old_base
            )
            propagation_residual = self._bounded_component(
                propagation_anomaly, self.propagation_cap_multiplier
            )
        else:
            alpha = torch.zeros_like(wind_prediction)
            propagation_residual = torch.zeros_like(wind_prediction)

        correction = self._bounded_component(
            raw_correction, self.correction_cap_multiplier
        )
        if self.training and self.hybrid_correction_drop_prob > 0.0:
            keep = (
                torch.rand(
                    correction.shape[0], 1, device=correction.device
                )
                >= self.hybrid_correction_drop_prob
            ).to(dtype=correction.dtype)
            correction = correction * keep

        prediction = wind_prediction + propagation_residual + correction

        source_speed = self.last_source_speed_kms
        arrival = self.last_arrival_hours
        self._last_hybrid_diagnostics = {
            "source_speed_mean_kms": (
                source_speed.mean() if source_speed is not None else wind.new_tensor(0.0)
            ),
            "source_speed_std_kms": (
                source_speed.std() if source_speed is not None else wind.new_tensor(0.0)
            ),
            "arrival_mean_h": (
                arrival.mean() if arrival is not None else wind.new_tensor(0.0)
            ),
            "arrival_std_h": (
                arrival.std() if arrival is not None else wind.new_tensor(0.0)
            ),
            "fusion_alpha": alpha.detach().mean(),
            "wind_residual_rms_kms": (
                wind_residual.detach().float().square().mean().sqrt() * 1000.0
            ),
            "propagation_residual_rms_kms": (
                propagation_residual.detach().float().square().mean().sqrt()
                * 1000.0
            ),
            "correction_rms_kms": (
                correction.detach().float().square().mean().sqrt() * 1000.0
            ),
            "delta_rms": difference.detach().float().square().mean().sqrt(),
        }
        if self.last_surge_prob is not None:
            self._last_hybrid_diagnostics["surge_probability"] = (
                self.last_surge_prob.mean()
            )

        components = {
            "ar_baseline": ar_baseline,
            "wind_prediction": wind_prediction,
            "wind_residual": wind_residual,
            "propagation_residual": propagation_residual,
            "correction": correction,
            "fusion_alpha": alpha,
        }
        aux = {
            **parent_aux,
            **components,
        }
        if return_components and return_aux:
            return prediction, components, aux
        if return_components:
            return prediction, components
        if return_aux:
            return prediction, aux
        return prediction

    def training_diagnostics(self):
        return self._last_hybrid_diagnostics
