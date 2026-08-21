import os


ENVIRONMENT_SUFFIXES = (
    "GRID_ROWS",
    "GRID_COLUMNS",
    "UNET_CHANNELS",
    "D_MODEL",
    "ATTENTION_HEADS",
    "DECODER_LAYERS",
    "FEEDFORWARD_DIM",
    "DROPOUT",
    "DELTA_GAIN",
    "TIME_MASK_PROBABILITY",
    "MODALITY_DROP_PROBABILITY",
    "TIMING_SIGMA_HOURS",
    "PHYSICAL_PRIOR_MIN",
    "PHYSICAL_PRIOR_MAX",
    "PHYSICAL_PRIOR_INIT",
    "MAXIMUM_BLEND",
    "INITIAL_BLEND",
    "CORRECTION_CAP_MULTIPLIER",
    "SOLAR_DISK_EDGE_PIXELS",
    "HINDCAST_WEIGHT_START",
    "HINDCAST_WEIGHT_END",
    "HINDCAST_DECAY_EPOCHS",
    "ALIGNMENT_WEIGHT",
    "ALIGNMENT_SIGMA_DEG",
    "CORRECTION_L2_WEIGHT",
    "GATE_L1_WEIGHT",
    "SPEED_SMOOTHNESS_WEIGHT",
    "GRADIENT_CLIP",
    "WEIGHT_DECAY",
    "WARMUP_EPOCHS",
    "MIN_LR",
    "EARLY_STOP_PATIENCE",
)


def map_v14_environment():
    for suffix in ENVIRONMENT_SUFFIXES:
        source = f"V14_{suffix}"
        if source in os.environ:
            os.environ[f"V13_{suffix}"] = os.environ[source]


map_v14_environment()

import train_solar_timing_transformer_v13 as trainer
from model_solar_deformable_timing_v14 import (
    ARCHITECTURE_NAME,
    FILE_STEM,
    SolarWindDeformableTimingV14,
)


FEATURE_SCHEMA = (
    "disk_mask_signed_delta_lite_unet_speed_locked_deformable_timing_v14"
)
CHECKPOINT_VERSION = "14"
_BASE_BUILD_MODEL_KWARGS = trainer.build_model_kwargs


def deformable_preprocess():
    return {
        "deformable_points": int(os.getenv("V14_DEFORMABLE_POINTS", "8")),
        "maximum_time_offset_hours": float(
            os.getenv("V14_MAXIMUM_TIME_OFFSET_HOURS", "12")
        ),
        "maximum_longitude_offset_cells": float(
            os.getenv("V14_MAXIMUM_LONGITUDE_OFFSET_CELLS", "1.5")
        ),
        "dense_kernel_time_frames": float(
            os.getenv("V14_DENSE_KERNEL_TIME_FRAMES", "0.75")
        ),
        "dense_kernel_longitude_cells": float(
            os.getenv("V14_DENSE_KERNEL_LONGITUDE_CELLS", "0.75")
        ),
    }


def build_model_kwargs(ar_fit, ar_residual_scale, wind_only=False):
    kwargs = _BASE_BUILD_MODEL_KWARGS(
        ar_fit, ar_residual_scale, wind_only=wind_only
    )
    kwargs.update(deformable_preprocess())
    return kwargs


def configure_trainer():
    trainer.ARCHITECTURE_NAME = ARCHITECTURE_NAME
    trainer.FILE_STEM = FILE_STEM
    trainer.FEATURE_SCHEMA = FEATURE_SCHEMA
    trainer.CHECKPOINT_VERSION = CHECKPOINT_VERSION
    trainer.MODEL_CLASS = SolarWindDeformableTimingV14
    trainer.SOURCE_MODELS = [
        "SolarWindTimingTransformerV13 speed-locked U-Net source map",
        "physics-guided sparse deformable attention",
    ]
    trainer.MODEL_CHANGES = [
        "top-K physical arrival references for each query and attention head",
        "learned bounded offsets only in acquisition time and longitude",
        "one unshifted physical anchor retained in every sparse attention set",
        "sampled source speed remains both transit controller and forecast value",
        "strict query-time causal clipping after every learned offset",
        "dense differentiable attention reconstruction for backmapping diagnostics",
    ]
    trainer.MANIFEST_VERSION_KEY = "v14_changes"
    trainer.EXTRA_PREPROCESS = deformable_preprocess()
    trainer.build_model_kwargs = build_model_kwargs


def main():
    map_v14_environment()
    configure_trainer()
    trainer.main()


if __name__ == "__main__":
    main()
