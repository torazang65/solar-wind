import os

from model_solar_cnn_v6 import SolarWindCNNTransformerV6
from train_solar_physics_v5 import main


if __name__ == "__main__":
    main(
        model_class=SolarWindCNNTransformerV6,
        architecture_name="SolarWindCNNTransformerV6",
        version=6,
        file_stem="solar_cnn_v6",
        feature_schema="v3_cea_cnn_relative_darkness_v1",
        extra_model_kwargs={
            "visual_dropout": float(os.getenv("SOLAR_VISUAL_DROPOUT", "0.10"))
        },
    )
