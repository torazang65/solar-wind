import os

from model_solar_arrival_v9 import SolarWindArrivalTCNV9
from train_solar_physics_v5 import main


if __name__ == "__main__":
    ema_decay = float(os.getenv("SOLAR_V9_EMA_DECAY", "0"))
    main(
        model_class=SolarWindArrivalTCNV9,
        architecture_name="SolarWindArrivalTCNV9",
        version=9,
        file_stem="solar_arrival_v9",
        feature_schema="cea_cnn_separate_tcn_learned_arrival_gate_v1",
        extra_model_kwargs={
            "image_cnn_channels": int(
                os.getenv("SOLAR_V9_IMAGE_CNN_CHANNELS", "48")
            ),
            "temporal_layers": int(os.getenv("SOLAR_V9_TEMPORAL_LAYERS", "3")),
            "visual_dropout": float(os.getenv("SOLAR_VISUAL_DROPOUT", "0.10")),
            "image_time_mask_probability": float(
                os.getenv("SOLAR_V9_TIME_MASK_PROBABILITY", "0.10")
            ),
            "image_modality_drop_probability": float(
                os.getenv("SOLAR_V9_MODALITY_DROP_PROBABILITY", "0.10")
            ),
            "transit_min_hours": float(
                os.getenv("SOLAR_V9_TRANSIT_MIN_HOURS", "48")
            ),
            "transit_max_hours": float(
                os.getenv("SOLAR_V9_TRANSIT_MAX_HOURS", "120")
            ),
            "arrival_sigma_hours": float(
                os.getenv("SOLAR_V9_ARRIVAL_SIGMA_HOURS", "24")
            ),
            "arrival_prior_strength": float(
                os.getenv("SOLAR_V9_ARRIVAL_PRIOR_STRENGTH", "0.25")
            ),
            "residual_cap_multiplier": float(
                os.getenv("SOLAR_V9_RESIDUAL_CAP_MULTIPLIER", "2.5")
            ),
        },
        training_image_flip_probability=0.0,
        residual_l2_weight=float(
            os.getenv("SOLAR_V9_RESIDUAL_L2_WEIGHT", "0.002")
        ),
        ema_decay=ema_decay if ema_decay > 0.0 else None,
        use_baseline_residual_scale=True,
    )
