import os

import torch
from torch.nn import functional as F

import train_solar_deformable_timing_v14 as v14_trainer


V14_ONLY_SUFFIXES = (
    "DEFORMABLE_POINTS",
    "MAXIMUM_TIME_OFFSET_HOURS",
    "MAXIMUM_LONGITUDE_OFFSET_CELLS",
    "DENSE_KERNEL_TIME_FRAMES",
    "DENSE_KERNEL_LONGITUDE_CELLS",
)


def map_v15_environment():
    for suffix in v14_trainer.ENVIRONMENT_SUFFIXES + V14_ONLY_SUFFIXES:
        source = f"V15_{suffix}"
        if source in os.environ:
            os.environ[f"V14_{suffix}"] = os.environ[source]
    v14_trainer.map_v14_environment()


map_v15_environment()

import train_solar_timing_transformer_v13 as trainer
from model_solar_peak_event_v15 import (
    ARCHITECTURE_NAME,
    FILE_STEM,
    SolarWindPeakEventV15,
)


FEATURE_SCHEMA = (
    "disk_mask_signed_delta_unet_deformable_direct_peak_time_value_v15"
)
CHECKPOINT_VERSION = "15"
_BASE_BUILD_MODEL_KWARGS = v14_trainer.build_model_kwargs


def peak_preprocess():
    preprocess = v14_trainer.deformable_preprocess()
    preprocess.update(
        {
            "peak_hidden_dim": int(os.getenv("V15_PEAK_HIDDEN_DIM", "96")),
            "peak_curve_sigma_steps": float(
                os.getenv("V15_PEAK_CURVE_SIGMA_STEPS", "1.25")
            ),
            "peak_value_bounds": [
                float(os.getenv("V15_PEAK_VALUE_MIN", "0.25")),
                float(os.getenv("V15_PEAK_VALUE_MAX", "0.90")),
            ],
            "maximum_peak_blend": float(
                os.getenv("V15_MAXIMUM_PEAK_BLEND", "0.30")
            ),
            "initial_peak_blend": float(
                os.getenv("V15_INITIAL_PEAK_BLEND", "0.05")
            ),
            "peak_correction_cap_multiplier": float(
                os.getenv("V15_PEAK_CORRECTION_CAP_MULTIPLIER", "1.0")
            ),
            "peak_time_loss_weight": float(
                os.getenv("V15_PEAK_TIME_LOSS_WEIGHT", "0.05")
            ),
            "peak_value_loss_weight": float(
                os.getenv("V15_PEAK_VALUE_LOSS_WEIGHT", "0.25")
            ),
            "peak_label_sigma_steps": float(
                os.getenv("V15_PEAK_LABEL_SIGMA_STEPS", "1.0")
            ),
            "peak_timing_prominence_kms": float(
                os.getenv("V15_PEAK_TIMING_PROMINENCE_KMS", "60")
            ),
            "peak_timing_minimum_weight": float(
                os.getenv("V15_PEAK_TIMING_MINIMUM_WEIGHT", "0.10")
            ),
        }
    )
    return preprocess


def build_model_kwargs(ar_fit, ar_residual_scale, wind_only=False):
    kwargs = _BASE_BUILD_MODEL_KWARGS(
        ar_fit, ar_residual_scale, wind_only=wind_only
    )
    peak_value_bounds = peak_preprocess()["peak_value_bounds"]
    kwargs.update(
        {
            "peak_hidden_dim": int(os.getenv("V15_PEAK_HIDDEN_DIM", "96")),
            "peak_curve_sigma_steps": float(
                os.getenv("V15_PEAK_CURVE_SIGMA_STEPS", "1.25")
            ),
            "peak_value_min": peak_value_bounds[0],
            "peak_value_max": peak_value_bounds[1],
            "maximum_peak_blend": float(
                os.getenv("V15_MAXIMUM_PEAK_BLEND", "0.30")
            ),
            "initial_peak_blend": float(
                os.getenv("V15_INITIAL_PEAK_BLEND", "0.05")
            ),
            "peak_correction_cap_multiplier": float(
                os.getenv("V15_PEAK_CORRECTION_CAP_MULTIPLIER", "1.0")
            ),
        }
    )
    return kwargs


def peak_event_objective(model, prediction, components, aux, wind, target):
    del model, prediction, components, wind
    target_peak_value, target_peak_index = target.max(dim=-1)
    target_minimum = target.min(dim=-1).values
    positions = torch.arange(
        target.shape[-1], device=target.device, dtype=target.dtype
    )
    label_sigma = float(os.getenv("V15_PEAK_LABEL_SIGMA_STEPS", "1.0"))
    target_distribution = torch.exp(
        -(
            positions.unsqueeze(0)
            - target_peak_index.to(dtype=target.dtype).unsqueeze(-1)
        ).square()
        / (2.0 * label_sigma**2)
    )
    target_distribution = target_distribution / target_distribution.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    timing_nll = -(
        target_distribution * F.log_softmax(aux["peak_time_logits"], dim=-1)
    ).sum(dim=-1)
    prominence_kms = (target_peak_value - target_minimum) * 1000.0
    prominence_scale = float(
        os.getenv("V15_PEAK_TIMING_PROMINENCE_KMS", "60")
    )
    minimum_weight = float(
        os.getenv("V15_PEAK_TIMING_MINIMUM_WEIGHT", "0.10")
    )
    timing_weight = (prominence_kms / prominence_scale).clamp(
        minimum_weight, 1.0
    )
    peak_time_loss = (timing_nll * timing_weight).sum() / timing_weight.sum()
    peak_value_loss = torch.sqrt(
        F.mse_loss(aux["peak_value"], target_peak_value) + 1e-8
    )
    time_weight = float(os.getenv("V15_PEAK_TIME_LOSS_WEIGHT", "0.05"))
    value_weight = float(os.getenv("V15_PEAK_VALUE_LOSS_WEIGHT", "0.25"))
    weighted_loss = time_weight * peak_time_loss + value_weight * peak_value_loss

    predicted_peak_index = aux["peak_time_probability"].argmax(dim=-1)
    time_error_h = (predicted_peak_index - target_peak_index).abs().float() * 6.0
    peak_value_error_kms = (
        aux["peak_value"] - target_peak_value
    ).float() * 1000.0
    return weighted_loss, {
        "peak_objective": weighted_loss,
        "peak_time_loss": peak_time_loss,
        "peak_time_mae_h": time_error_h.mean(),
        "peak_time_within_6h": (time_error_h <= 6.0).float().mean(),
        "peak_value_rmse_km_s": torch.sqrt(
            peak_value_error_kms.square().mean() + 1e-8
        ),
        "target_peak_prominence_km_s": prominence_kms.mean(),
    }


def configure_trainer():
    v14_trainer.configure_trainer()
    trainer.ARCHITECTURE_NAME = ARCHITECTURE_NAME
    trainer.FILE_STEM = FILE_STEM
    trainer.FEATURE_SCHEMA = FEATURE_SCHEMA
    trainer.CHECKPOINT_VERSION = CHECKPOINT_VERSION
    trainer.MODEL_CLASS = SolarWindPeakEventV15
    trainer.SOURCE_MODELS = [
        "SolarWindDeformableTimingV14 physical timing backbone",
        "direct future peak-time and peak-value supervision",
    ]
    trainer.MODEL_CHANGES = [
        "derive a 12-bin peak-time label directly from each future target",
        "downweight ambiguous timing labels when future peak prominence is low",
        "predict absolute peak speed with a separate bounded value head",
        "inject only a capped Gaussian event correction around predicted time",
        "event path bypasses V13 evidence collapse but remains image-mask gated",
        "image-off prediction remains exactly the train-only AR(2) fallback",
    ]
    trainer.MANIFEST_VERSION_KEY = "v15_changes"
    trainer.EXTRA_PREPROCESS = peak_preprocess()
    trainer.AUXILIARY_OBJECTIVE = peak_event_objective
    trainer.build_model_kwargs = build_model_kwargs


def main():
    map_v15_environment()
    configure_trainer()
    trainer.main()


if __name__ == "__main__":
    main()
