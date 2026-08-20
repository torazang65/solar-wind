import os

from model_baseline_v2_4 import SolarWindFixedLagMagnitudeTransformerV24
from train_baseline_v2_3 import fit_ar_configuration
from train_solar_physics_v5 import main


if __name__ == "__main__":
    ar_fit, ar_residual_scale = fit_ar_configuration(
        order_environment="V24_AR_ORDER",
        ridge_environment="V24_AR_RIDGE",
    )
    ema_decay = float(os.getenv("V24_EMA_DECAY", "0.995"))
    main(
        model_class=SolarWindFixedLagMagnitudeTransformerV24,
        architecture_name="SolarWindFixedLagMagnitudeTransformerV24",
        version="2.4",
        file_stem="baseline_v2_4",
        feature_schema="global_ar2_fixed_96h_lag_attentive_scale_v1",
        extra_model_kwargs={
            "ar_coefficients": ar_fit.coefficients.tolist(),
            "ar_intercept": ar_fit.intercept,
            "ar_ridge_strength": ar_fit.ridge_strength,
            "baseline_residual_scale": ar_residual_scale.tolist(),
            "wind_residual_cap_multiplier": float(
                os.getenv("V24_WIND_RESIDUAL_CAP_MULTIPLIER", "0.75")
            ),
            "fixed_lag_hours": float(os.getenv("V24_FIXED_LAG_HOURS", "96")),
            "fixed_lag_sigma_hours": float(
                os.getenv("V24_FIXED_LAG_SIGMA_HOURS", "12")
            ),
            "fixed_lag_window_hours": float(
                os.getenv("V24_FIXED_LAG_WINDOW_HOURS", "24")
            ),
            "local_temporal_radius_hours": float(
                os.getenv("V24_LOCAL_TEMPORAL_RADIUS_HOURS", "12")
            ),
            "central_spatial_prior_strength": float(
                os.getenv("V24_CENTRAL_SPATIAL_PRIOR_STRENGTH", "0.5")
            ),
            "image_scale_limit": float(
                os.getenv("V24_IMAGE_SCALE_LIMIT", "0.15")
            ),
            "delta_gain": float(os.getenv("V24_DELTA_GAIN", "4.0")),
            "image_time_mask_probability": float(
                os.getenv("V24_TIME_MASK_PROBABILITY", "0.05")
            ),
            "image_modality_drop_probability": float(
                os.getenv("V24_MODALITY_DROP_PROBABILITY", "0.20")
            ),
            "timing_prior_strength": 0.0,
            "timing_prior_sigma_hours": float(
                os.getenv("V24_FIXED_LAG_SIGMA_HOURS", "12")
            ),
            "residual_cap_multiplier": float(
                os.getenv("V24_IMAGE_RESIDUAL_CAP_MULTIPLIER", "1.0")
            ),
        },
        training_image_flip_probability=0.0,
        residual_l2_weight=float(
            os.getenv("V24_RESIDUAL_L2_WEIGHT", "0.002")
        ),
        ema_decay=ema_decay if ema_decay > 0.0 else None,
        use_baseline_residual_scale=False,
        scheduler_kind="warmup_cosine",
        warmup_epochs=int(os.getenv("V24_WARMUP_EPOCHS", "3")),
        minimum_learning_rate=float(os.getenv("V24_MIN_LR", "1e-6")),
        optimizer_weight_decay=float(os.getenv("V24_WEIGHT_DECAY", "0.02")),
        early_stopping_patience=int(
            os.getenv("V24_EARLY_STOP_PATIENCE", "15")
        ),
        grid_label="spatial_grid",
    )
