import os

from model_solar_ballistic_v8 import SolarWindBallisticTransformerV8
from train_solar_physics_v5 import main


if __name__ == "__main__":
    ema_decay = float(os.getenv("SOLAR_V8_EMA_DECAY", "0.995"))
    main(
        model_class=SolarWindBallisticTransformerV8,
        architecture_name="SolarWindBallisticTransformerV8",
        version=8,
        file_stem="solar_ballistic_v8",
        feature_schema="v6_cea_cnn_ballistic_attention_v1",
        extra_model_kwargs={
            "visual_dropout": float(os.getenv("SOLAR_VISUAL_DROPOUT", "0.10")),
            "physics_prior_strength": float(
                os.getenv("SOLAR_V8_PHYSICS_PRIOR_STRENGTH", "1.0")
            ),
            "longitude_sigma_degrees": float(
                os.getenv("SOLAR_V8_LONGITUDE_SIGMA_DEGREES", "30.0")
            ),
            "latitude_sigma_degrees": float(
                os.getenv("SOLAR_V8_LATITUDE_SIGMA_DEGREES", "45.0")
            ),
            "residual_cap_multiplier": float(
                os.getenv("SOLAR_V8_RESIDUAL_CAP_MULTIPLIER", "2.5")
            ),
        },
        training_image_flip_probability=float(
            os.getenv("SOLAR_V8_NORTH_SOUTH_FLIP_PROBABILITY", "0.5")
        ),
        residual_l2_weight=float(os.getenv("SOLAR_V8_RESIDUAL_L2_WEIGHT", "0.01")),
        ema_decay=ema_decay if ema_decay > 0.0 else None,
        use_baseline_residual_scale=True,
    )
