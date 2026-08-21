import os


V12_SUFFIXES = (
    "LAG_HOURS",
    "LAG_SIGMA_HOURS",
    "LAG_PRIOR_MAX_STRENGTH",
    "LAG_PRIOR_INIT_STRENGTH",
    "FRAME_DIM",
    "LSTM_HIDDEN_DIM",
    "LSTM_LAYERS",
    "WIND_FEATURE_DIM",
    "DROPOUT",
    "TIME_MASK_PROBABILITY",
    "MODALITY_DROP_PROBABILITY",
    "DELTA_GAIN",
    "WIND_RESIDUAL_CAP_MULTIPLIER",
    "IMAGE_CORRECTION_CAP_MULTIPLIER",
    "SOLAR_DISK_EDGE_PIXELS",
    "AR_ORDER",
    "AR_RIDGE",
    "WEIGHT_DECAY",
    "WARMUP_EPOCHS",
    "MIN_LR",
    "WIND_AUX_WEIGHT",
    "LAG_ALIGNMENT_WEIGHT",
    "ALIGNMENT_SIGMA_HOURS",
    "CORRECTION_L2_WEIGHT",
    "GRADIENT_CLIP",
    "EARLY_STOP_PATIENCE",
)


def map_v16_environment():
    for suffix in V12_SUFFIXES:
        source = f"V16_{suffix}"
        if source in os.environ:
            os.environ[f"V12_{suffix}"] = os.environ[source]
    os.environ["V12_GRID_ROWS"] = "1"
    os.environ["V12_GRID_COLUMNS"] = os.getenv("IMAGE_SIZE", "64")


map_v16_environment()

import train_solar_lstm_v12 as trainer
from model_solar_native_profile_lstm_v16 import (
    ARCHITECTURE_NAME,
    FILE_STEM,
    SolarWindNativeProfileLSTMV16,
)


FEATURE_SCHEMA = "native_64_column_stats_delta_lstm_fixed_lag_v16"
CHECKPOINT_VERSION = "16"
_BASE_BUILD_MODEL_KWARGS = trainer.build_model_kwargs


def boolean_environment(name, default=False):
    fallback = "1" if default else "0"
    return os.getenv(name, fallback).lower() in {"1", "true", "yes"}


def native_preprocess():
    return {
        "image_encoder": "native_longitude_profile",
        "native_longitude_columns": int(os.getenv("IMAGE_SIZE", "64")),
        "column_statistics": ["mean", "min", "max", "std"],
        "signed_profile_differences": True,
        "column_dim": int(os.getenv("V16_COLUMN_DIM", "16")),
        "longitude_kernel_size": int(
            os.getenv("V16_LONGITUDE_KERNEL_SIZE", "5")
        ),
        "scramble_images": boolean_environment("V16_SCRAMBLE_IMAGES"),
        "disable_wind_residual": boolean_environment(
            "V16_DISABLE_WIND_RESIDUAL", True
        ),
    }


def build_model_kwargs(ar_fit, ar_residual_scale, wind_only=False):
    kwargs = _BASE_BUILD_MODEL_KWARGS(
        ar_fit, ar_residual_scale, wind_only=wind_only
    )
    kwargs.update(
        {
            "column_dim": int(os.getenv("V16_COLUMN_DIM", "16")),
            "longitude_kernel_size": int(
                os.getenv("V16_LONGITUDE_KERNEL_SIZE", "5")
            ),
            "scramble_images": boolean_environment("V16_SCRAMBLE_IMAGES"),
            "disable_wind_residual": boolean_environment(
                "V16_DISABLE_WIND_RESIDUAL", True
            ),
        }
    )
    return kwargs


def configure_trainer():
    trainer.ARCHITECTURE_NAME = ARCHITECTURE_NAME
    trainer.FILE_STEM = FILE_STEM
    trainer.FEATURE_SCHEMA = FEATURE_SCHEMA
    trainer.MODEL_CLASS = SolarWindNativeProfileLSTMV16
    trainer.CHECKPOINT_VERSION = CHECKPOINT_VERSION
    trainer.MANIFEST_VERSION_KEY = "v16_changes"
    trainer.MODEL_CHANGES = [
        "no image resize, learned pooling, U-Net, or spatial pyramid",
        "native-width per-longitude mean/min/max/std statistics",
        "signed temporal differences of every native column statistic",
        "stride-one longitude convolution and one-layer LSTM",
        "fixed-lag attention with train-only AR(2) anchor",
        "frozen neural wind residual for an exact AR wind baseline",
        "matched scrambled-image and wind-only controls",
    ]
    trainer.EXTRA_PREPROCESS = native_preprocess()
    trainer.build_model_kwargs = build_model_kwargs


def main():
    map_v16_environment()
    configure_trainer()
    trainer.main()


if __name__ == "__main__":
    main()
