import os

from model_baseline_v2_2 import SolarWindBaselineSpatialTransformerV22
from train_solar_physics_v5 import main


if __name__ == "__main__":
    ema_decay = float(os.getenv("V22_EMA_DECAY", "0.995"))
    main(
        model_class=SolarWindBaselineSpatialTransformerV22,
        architecture_name="SolarWindBaselineSpatialTransformerV22",
        version="2.2",
        file_stem="baseline_v2_2",
        feature_schema="masked_intensity_delta_full_4x4_tokens_v2",
        extra_model_kwargs={
            "delta_gain": float(os.getenv("V22_DELTA_GAIN", "4.0")),
            "image_time_mask_probability": float(
                os.getenv("V22_TIME_MASK_PROBABILITY", "0.15")
            ),
            "image_modality_drop_probability": float(
                os.getenv("V22_MODALITY_DROP_PROBABILITY", "0.25")
            ),
            "timing_prior_strength": float(
                os.getenv("V22_TIMING_PRIOR_STRENGTH", "0.0")
            ),
            "timing_prior_sigma_hours": float(
                os.getenv("V22_TIMING_PRIOR_SIGMA_HOURS", "36")
            ),
            "residual_cap_multiplier": float(
                os.getenv("V22_RESIDUAL_CAP_MULTIPLIER", "1.5")
            ),
        },
        training_image_flip_probability=0.0,
        residual_l2_weight=float(os.getenv("V22_RESIDUAL_L2_WEIGHT", "0.002")),
        ema_decay=ema_decay if ema_decay > 0.0 else None,
        use_baseline_residual_scale=True,
        scheduler_kind="warmup_cosine",
        warmup_epochs=int(os.getenv("V22_WARMUP_EPOCHS", "3")),
        minimum_learning_rate=float(os.getenv("V22_MIN_LR", "1e-6")),
        optimizer_weight_decay=float(os.getenv("V22_WEIGHT_DECAY", "0.02")),
        early_stopping_patience=int(os.getenv("V22_EARLY_STOP_PATIENCE", "15")),
        grid_label="spatial_grid",
    )
