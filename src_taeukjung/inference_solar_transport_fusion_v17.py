from train_solar_transport_fusion_v17 import (
    CHECKPOINT_VERSION,
    FEATURE_SCHEMA,
    v17_preprocess,
)

import inference_solar_lstm_v12 as inference
from model_solar_transport_fusion_v17 import (
    ARCHITECTURE_NAME,
    FILE_STEM,
    SolarWindTransportFusionV17,
)


def current_preprocess():
    return v17_preprocess()


def configure_inference():
    inference.ARCHITECTURE_NAME = ARCHITECTURE_NAME
    inference.FILE_STEM = FILE_STEM
    inference.FEATURE_SCHEMA = FEATURE_SCHEMA
    inference.MODEL_CLASS = SolarWindTransportFusionV17
    inference.CHECKPOINT_VERSION = CHECKPOINT_VERSION
    inference.current_preprocess = current_preprocess


def main():
    configure_inference()
    inference.main()


if __name__ == "__main__":
    main()
