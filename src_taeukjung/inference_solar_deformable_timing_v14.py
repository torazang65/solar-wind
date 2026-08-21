from train_solar_deformable_timing_v14 import (
    CHECKPOINT_VERSION,
    FEATURE_SCHEMA,
    deformable_preprocess,
    map_v14_environment,
)


map_v14_environment()

import inference_solar_timing_transformer_v13 as inference
from model_solar_deformable_timing_v14 import (
    ARCHITECTURE_NAME,
    FILE_STEM,
    SolarWindDeformableTimingV14,
)


def configure_inference():
    inference.ARCHITECTURE_NAME = ARCHITECTURE_NAME
    inference.FILE_STEM = FILE_STEM
    inference.FEATURE_SCHEMA = FEATURE_SCHEMA
    inference.CHECKPOINT_VERSION = CHECKPOINT_VERSION
    inference.MODEL_CLASS = SolarWindDeformableTimingV14
    inference.EXTRA_PREPROCESS = deformable_preprocess()


def main():
    map_v14_environment()
    configure_inference()
    inference.main()


if __name__ == "__main__":
    main()
