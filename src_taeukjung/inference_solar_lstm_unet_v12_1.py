import inference_solar_lstm_v12 as inference
from model_solar_lstm_unet_v12_1 import (
    ARCHITECTURE_NAME,
    FILE_STEM,
    SolarWindLagLSTMUNetV12_1,
)
from train_solar_lstm_unet_v12_1 import (
    CHECKPOINT_VERSION,
    FEATURE_SCHEMA,
    parse_unet_channels,
)


def configure_inference():
    inference.ARCHITECTURE_NAME = ARCHITECTURE_NAME
    inference.FILE_STEM = FILE_STEM
    inference.FEATURE_SCHEMA = FEATURE_SCHEMA
    inference.MODEL_CLASS = SolarWindLagLSTMUNetV12_1
    inference.CHECKPOINT_VERSION = CHECKPOINT_VERSION
    inference.EXTRA_PREPROCESS = {
        "image_encoder": "lite_unet",
        "unet_channels": list(parse_unet_channels()),
    }


def main():
    configure_inference()
    inference.main()


if __name__ == "__main__":
    main()
