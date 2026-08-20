import os

from model_solar_factorized_v7 import SolarWindFactorizedTransformerV7
from train_solar_physics_v5 import main


if __name__ == "__main__":
    main(
        model_class=SolarWindFactorizedTransformerV7,
        architecture_name="SolarWindFactorizedTransformerV7",
        version=7,
        file_stem="solar_factorized_v7",
        feature_schema="v3_cea_cnn_factorized_attention_v1",
        extra_model_kwargs={
            "visual_dropout": float(os.getenv("SOLAR_VISUAL_DROPOUT", "0.10"))
        },
    )
