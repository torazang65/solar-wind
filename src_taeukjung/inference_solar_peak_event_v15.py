from train_solar_peak_event_v15 import (
    CHECKPOINT_VERSION,
    FEATURE_SCHEMA,
    map_v15_environment,
    peak_preprocess,
)


map_v15_environment()

import inference_solar_timing_transformer_v13 as inference
from model_solar_peak_event_v15 import (
    ARCHITECTURE_NAME,
    FILE_STEM,
    SolarWindPeakEventV15,
)


def configure_inference():
    inference.ARCHITECTURE_NAME = ARCHITECTURE_NAME
    inference.FILE_STEM = FILE_STEM
    inference.FEATURE_SCHEMA = FEATURE_SCHEMA
    inference.CHECKPOINT_VERSION = CHECKPOINT_VERSION
    inference.MODEL_CLASS = SolarWindPeakEventV15
    inference.EXTRA_PREPROCESS = peak_preprocess()


def main():
    map_v15_environment()
    configure_inference()
    inference.main()


if __name__ == "__main__":
    main()
