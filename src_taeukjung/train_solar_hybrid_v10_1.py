import os

from model_solar_hybrid_v10_1 import (
    ARCHITECTURE_NAME,
    FILE_STEM,
    SolarWindSelectiveHybridV101,
)
from train_solar_hybrid_v10 import main


FEATURE_SCHEMA = (
    "seokho_v5b_masked_delta4_ar2_fixed96h_propagation_"
    "surge_fast_selective_correction_v1"
)


if __name__ == "__main__":
    main(
        model_class=SolarWindSelectiveHybridV101,
        architecture_name=ARCHITECTURE_NAME,
        version="10.1",
        file_stem=FILE_STEM,
        feature_schema=FEATURE_SCHEMA,
        extra_model_kwargs={
            "wind_residual_mix": float(
                os.getenv("V101_WIND_RESIDUAL_MIX", "0.0")
            ),
            "correction_min_gate": float(
                os.getenv("V101_CORRECTION_MIN_GATE", "0.15")
            ),
            "correction_surge_power": float(
                os.getenv("V101_CORRECTION_SURGE_POWER", "1.0")
            ),
            "fast_wind_threshold_kms": float(
                os.getenv("V101_FAST_WIND_THRESHOLD_KMS", "550")
            ),
            "fast_wind_scale_kms": float(
                os.getenv("V101_FAST_WIND_SCALE_KMS", "50")
            ),
            "fast_quiet_suppression": float(
                os.getenv("V101_FAST_QUIET_SUPPRESSION", "0.75")
            ),
        },
    )
