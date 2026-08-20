import os

import numpy as np

from ar_wind import (
    fit_global_ar,
    predict_recursive_ar,
    residual_scale,
    validation_metrics,
)
from chain_sampling import infer_temporal_chains
from dataset import (
    IMAGE_COLUMNS,
    WIND_COLUMNS,
    train_inputs,
    train_targets,
    val_inputs,
    val_targets,
)
from model_baseline_v2_3 import SolarWindARNeuralTransformerV23
from train_solar_physics_v5 import main


def fit_ar_configuration(
    order_environment="V23_AR_ORDER",
    ridge_environment="V23_AR_RIDGE",
):
    order = int(os.getenv(order_environment, "2"))
    ridge_strength = float(os.getenv(ridge_environment, "30"))
    train_chains = infer_temporal_chains(train_inputs, IMAGE_COLUMNS)
    val_chains = infer_temporal_chains(val_inputs, IMAGE_COLUMNS)
    fit = fit_global_ar(
        train_inputs,
        train_targets,
        train_chains,
        WIND_COLUMNS,
        order=order,
        ridge_strength=ridge_strength,
    )
    train_wind = train_inputs[WIND_COLUMNS].to_numpy(np.float64) / 1000.0
    val_wind = val_inputs[WIND_COLUMNS].to_numpy(np.float64) / 1000.0
    train_prediction = predict_recursive_ar(
        train_wind, fit.coefficients, fit.intercept
    )
    val_prediction = predict_recursive_ar(
        val_wind, fit.coefficients, fit.intercept
    )
    scale = residual_scale(train_targets, train_prediction)
    val_micro, val_macro = validation_metrics(
        val_targets, val_prediction, val_chains.chain_ids
    )
    print(
        f"global_arima_order=({fit.order},0,0) "
        f"ridge={fit.ridge_strength:.3f} transitions={fit.transition_count} "
        f"coefficients={np.round(fit.coefficients, 6).tolist()} "
        f"intercept={fit.intercept:.6f}"
    )
    print(
        f"global_ar_val_rmse={val_micro:.3f} "
        f"global_ar_chain_macro_rmse={val_macro:.3f} "
        f"ar_residual_scale_km_s={np.round(scale * 1000.0, 2).tolist()}"
    )
    return fit, scale


if __name__ == "__main__":
    ar_fit, ar_residual_scale = fit_ar_configuration()
    ema_decay = float(os.getenv("V23_EMA_DECAY", "0.995"))
    main(
        model_class=SolarWindARNeuralTransformerV23,
        architecture_name="SolarWindARNeuralTransformerV23",
        version="2.3",
        file_stem="baseline_v2_3",
        feature_schema="global_ar2_masked_intensity_delta_full_4x4_tokens_v1",
        extra_model_kwargs={
            "ar_coefficients": ar_fit.coefficients.tolist(),
            "ar_intercept": ar_fit.intercept,
            "ar_ridge_strength": ar_fit.ridge_strength,
            "baseline_residual_scale": ar_residual_scale.tolist(),
            "wind_residual_cap_multiplier": float(
                os.getenv("V23_WIND_RESIDUAL_CAP_MULTIPLIER", "0.75")
            ),
            "delta_gain": float(os.getenv("V23_DELTA_GAIN", "4.0")),
            "image_time_mask_probability": float(
                os.getenv("V23_TIME_MASK_PROBABILITY", "0.15")
            ),
            "image_modality_drop_probability": float(
                os.getenv("V23_MODALITY_DROP_PROBABILITY", "0.25")
            ),
            "timing_prior_strength": float(
                os.getenv("V23_TIMING_PRIOR_STRENGTH", "0.0")
            ),
            "timing_prior_sigma_hours": float(
                os.getenv("V23_TIMING_PRIOR_SIGMA_HOURS", "36")
            ),
            "residual_cap_multiplier": float(
                os.getenv("V23_IMAGE_RESIDUAL_CAP_MULTIPLIER", "1.5")
            ),
        },
        training_image_flip_probability=0.0,
        residual_l2_weight=float(os.getenv("V23_RESIDUAL_L2_WEIGHT", "0.002")),
        ema_decay=ema_decay if ema_decay > 0.0 else None,
        use_baseline_residual_scale=False,
        scheduler_kind="warmup_cosine",
        warmup_epochs=int(os.getenv("V23_WARMUP_EPOCHS", "3")),
        minimum_learning_rate=float(os.getenv("V23_MIN_LR", "1e-6")),
        optimizer_weight_decay=float(os.getenv("V23_WEIGHT_DECAY", "0.02")),
        early_stopping_patience=int(os.getenv("V23_EARLY_STOP_PATIENCE", "15")),
        grid_label="spatial_grid",
    )
