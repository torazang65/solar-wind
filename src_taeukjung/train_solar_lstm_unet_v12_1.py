import os

import train_solar_lstm_v12 as trainer
from model_solar_lstm_unet_v12_1 import (
    ARCHITECTURE_NAME,
    FILE_STEM,
    SolarWindLagLSTMUNetV12_1,
)


FEATURE_SCHEMA = "disk_mask_signed_delta_lite_unet_lstm_soft_lag_v12_1"
CHECKPOINT_VERSION = "12.1"


def parse_unet_channels(value=None):
    text = value if value is not None else os.getenv(
        "V12_1_UNET_CHANNELS", "12,16,24,40,56"
    )
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if len(values) != 5 or any(value <= 0 for value in values):
        raise ValueError("V12_1_UNET_CHANNELS must contain five positive integers")
    return values


base_build_model_kwargs = trainer.build_model_kwargs


def build_model_kwargs(ar_fit, ar_residual_scale, wind_only=False):
    kwargs = base_build_model_kwargs(
        ar_fit, ar_residual_scale, wind_only=wind_only
    )
    kwargs["unet_channels"] = list(parse_unet_channels())
    return kwargs


def main():
    unet_channels = list(parse_unet_channels())
    trainer.ARCHITECTURE_NAME = ARCHITECTURE_NAME
    trainer.FILE_STEM = FILE_STEM
    trainer.FEATURE_SCHEMA = FEATURE_SCHEMA
    trainer.MODEL_CLASS = SolarWindLagLSTMUNetV12_1
    trainer.CHECKPOINT_VERSION = CHECKPOINT_VERSION
    trainer.MANIFEST_VERSION_KEY = "v12_1_changes"
    trainer.MODEL_CHANGES = [
        "V12 AR(2), wind residual, LSTM, and soft lag back end retained",
        "partial U-Net encoder with 8x and 4x skip fusion",
        "decoder stopped at quarter resolution before 2x8 token pooling",
        "GroupNorm used for stable small-batch training",
    ]
    trainer.EXTRA_PREPROCESS = {
        "image_encoder": "lite_unet",
        "unet_channels": unet_channels,
    }
    trainer.build_model_kwargs = build_model_kwargs
    trainer.main()


if __name__ == "__main__":
    main()
