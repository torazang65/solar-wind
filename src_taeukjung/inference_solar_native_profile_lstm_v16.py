from train_solar_native_profile_lstm_v16 import (
    CHECKPOINT_VERSION,
    FEATURE_SCHEMA,
    map_v16_environment,
    native_preprocess,
)


map_v16_environment()

import inference_solar_lstm_v12 as inference
from model_solar_native_profile_lstm_v16 import (
    ARCHITECTURE_NAME,
    FILE_STEM,
    SolarWindNativeProfileLSTMV16,
)


def configure_inference():
    inference.ARCHITECTURE_NAME = ARCHITECTURE_NAME
    inference.FILE_STEM = FILE_STEM
    inference.FEATURE_SCHEMA = FEATURE_SCHEMA
    inference.MODEL_CLASS = SolarWindNativeProfileLSTMV16
    inference.CHECKPOINT_VERSION = CHECKPOINT_VERSION
    inference.EXTRA_PREPROCESS = native_preprocess()


def main():
    map_v16_environment()
    configure_inference()
    inference.main()


if __name__ == "__main__":
    main()
