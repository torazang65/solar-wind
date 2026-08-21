import json
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from ar_wind import (
    fit_global_ar,
    predict_recursive_ar,
    residual_scale,
    validation_metrics,
)
from chain_sampling import (
    ChainAwareSolarWindDataset,
    chain_manifest,
    infer_temporal_chains,
    make_chain_loader,
)
from config import *
from dataset import (
    IMAGE_COLUMNS,
    TARGET_COLUMNS,
    WIND_COLUMNS,
    train_image_array,
    train_image_index,
    train_index,
    train_inputs,
    train_targets,
    val_image_array,
    val_image_index,
    val_index,
    val_inputs,
    val_targets,
)
from model_solar_lstm_v12 import (
    ARCHITECTURE_NAME,
    FILE_STEM,
    FORECAST_STEPS,
    SolarWindLagLSTMV12,
)
from train_solar_hybrid_v10 import limited_indexes, save_validation_predictions


FEATURE_SCHEMA = "disk_mask_signed_delta_lon_preserving_cnn_lstm_soft_lag_v12"
HOURS_AT_1_AU_PER_1000_KMS = 149_597_870.7 / 1000.0 / 3600.0
MODEL_CLASS = SolarWindLagLSTMV12
CHECKPOINT_VERSION = "12"
MANIFEST_VERSION_KEY = "v12_changes"
MODEL_CHANGES = [
    "soft solar-disk mask and signed image differences",
    "longitude-preserving 2x8 CNN grid at 64 px",
    "full-sequence LSTM horizon attention",
    "multi-expert soft lag prior",
    "AR(2) plus full-history neural wind anchor",
    "bounded gated image correction",
]
EXTRA_PREPROCESS = {}


def parse_lag_hours(value=None):
    text = value if value is not None else os.getenv(
        "V12_LAG_HOURS", "72,84,96,108,120"
    )
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("V12_LAG_HOURS must not be empty")
    return values


def fit_ar_configuration(train_chains, val_chains):
    order = int(os.getenv("V12_AR_ORDER", "2"))
    ridge_strength = float(os.getenv("V12_AR_RIDGE", "30"))
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
    return fit, scale, val_micro, val_macro


def build_model_kwargs(ar_fit, ar_residual_scale, wind_only=False):
    lag_prior_max_strength = float(
        os.getenv("V12_LAG_PRIOR_MAX_STRENGTH", "2.0")
    )
    lag_prior_init_strength = min(
        float(os.getenv("V12_LAG_PRIOR_INIT_STRENGTH", "1.0")),
        lag_prior_max_strength,
    )
    return {
        "image_size": IMAGE_SIZE,
        "use_images": not wind_only,
        "ar_coefficients": ar_fit.coefficients.tolist(),
        "ar_intercept": ar_fit.intercept,
        "baseline_residual_scale": ar_residual_scale.tolist(),
        "grid_rows": int(os.getenv("V12_GRID_ROWS", "2")),
        "grid_columns": int(os.getenv("V12_GRID_COLUMNS", "8")),
        "cell_dim": int(os.getenv("V12_CELL_DIM", "48")),
        "frame_dim": int(os.getenv("V12_FRAME_DIM", "256")),
        "lstm_hidden_dim": int(os.getenv("V12_LSTM_HIDDEN_DIM", "192")),
        "lstm_layers": int(os.getenv("V12_LSTM_LAYERS", "1")),
        "wind_feature_dim": int(os.getenv("V12_WIND_FEATURE_DIM", "128")),
        "dropout": float(os.getenv("V12_DROPOUT", "0.15")),
        "time_mask_prob": float(
            os.getenv("V12_TIME_MASK_PROBABILITY", "0.10")
        ),
        "modality_drop_prob": float(
            os.getenv("V12_MODALITY_DROP_PROBABILITY", "0.15")
        ),
        "delta_gain": float(os.getenv("V12_DELTA_GAIN", "1.0")),
        "lag_hours": parse_lag_hours(),
        "lag_sigma_hours": float(os.getenv("V12_LAG_SIGMA_HOURS", "12")),
        "lag_prior_max_strength": lag_prior_max_strength,
        "lag_prior_init_strength": lag_prior_init_strength,
        "wind_residual_cap_multiplier": float(
            os.getenv("V12_WIND_RESIDUAL_CAP_MULTIPLIER", "1.0")
        ),
        "image_correction_cap_multiplier": float(
            os.getenv("V12_IMAGE_CORRECTION_CAP_MULTIPLIER", "2.0")
        ),
        "apply_solar_disk_mask": SOLAR_DISK_MASK,
        "solar_disk_center_fraction": SOLAR_DISK_CENTER_FRACTION,
        "solar_disk_radius_fraction": SOLAR_DISK_RADIUS_FRACTION,
        "solar_disk_edge_pixels": float(
            os.getenv("V12_SOLAR_DISK_EDGE_PIXELS", "1.5")
        ),
    }


def make_scheduler(optimizer, peak_lr):
    warmup_epochs = int(os.getenv("V12_WARMUP_EPOCHS", "3"))
    minimum_lr = float(os.getenv("V12_MIN_LR", "1e-6"))
    if EPOCHS == 1:
        return (
            torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
            0,
            minimum_lr,
        )
    if not 0 < warmup_epochs < EPOCHS:
        raise ValueError("V12_WARMUP_EPOCHS must be between 1 and EPOCHS - 1")

    def learning_rate_factor(step):
        if step < warmup_epochs:
            return (step + 1) / warmup_epochs
        progress = (step - warmup_epochs) / max(1, EPOCHS - warmup_epochs)
        floor = minimum_lr / peak_lr
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * cosine

    return (
        torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor),
        warmup_epochs,
        minimum_lr,
    )


def lag_alignment_kl(model, attention, target, time_keep, image_keep, sigma_hours):
    with torch.amp.autocast(AMP_DEVICE_TYPE, enabled=False):
        target = target.float().clamp(0.25, 0.90)
        transit_hours = HOURS_AT_1_AU_PER_1000_KMS / target
        transit_hours = transit_hours.clamp(
            float(model.lag_hours.min()), float(model.lag_hours.max())
        )
        delay = model.horizon_hours.float().view(1, FORECAST_STEPS, 1) + (
            model.image_age_hours.float().view(1, 1, -1)
        )
        target_logits = -(
            delay - transit_hours.unsqueeze(-1)
        ).square() / (2.0 * float(sigma_hours) ** 2)
        target_logits = target_logits.masked_fill(
            time_keep.float().unsqueeze(1) <= 0.0, -1e4
        )
        target_distribution = torch.softmax(target_logits, dim=-1)
        learned = attention.float().clamp_min(1e-8)
        kl = (
            target_distribution
            * (
                target_distribution.clamp_min(1e-8).log() - learned.log()
            )
        ).sum(dim=-1)
        valid = image_keep.float().unsqueeze(-1).expand_as(kl)
        return (kl * valid).sum() / valid.sum().clamp_min(1.0)


def run_epoch(
    model,
    loader,
    chain_count,
    optimizer,
    scaler,
    training,
    wind_aux_weight,
    alignment_weight,
    alignment_sigma_hours,
    correction_l2_weight,
    gradient_clip,
    collect_predictions=False,
):
    model.train(training)
    totals = {
        "forecast": 0.0,
        "wind_base": 0.0,
        "ar_base": 0.0,
        "correction": 0.0,
        "alignment": 0.0,
    }
    value_count = 0
    sample_count = 0
    chain_squared_error = np.zeros(chain_count, dtype=np.float64)
    chain_value_count = np.zeros(chain_count, dtype=np.int64)
    diagnostics_sum = {}
    predictions = []
    sample_ids = []
    chain_ids_output = []
    chain_positions_output = []

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["images"].to(DEVICE, non_blocking=PIN_MEMORY)
        wind = batch["wind"].to(DEVICE, non_blocking=PIN_MEMORY)
        target = batch["target"].to(DEVICE, non_blocking=PIN_MEMORY)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(AMP_DEVICE_TYPE, enabled=USE_AMP):
                prediction, components, aux = model(
                    images,
                    wind,
                    return_components=True,
                    return_aux=True,
                )
                forecast_loss = torch.sqrt(
                    F.mse_loss(prediction, target) + RMSE_EPSILON
                )
                wind_loss = torch.sqrt(
                    F.mse_loss(components["wind_base"], target)
                    + RMSE_EPSILON
                )
                alignment_value = lag_alignment_kl(
                    model,
                    aux["lag_attention"],
                    target,
                    aux["time_keep"],
                    aux["image_keep"],
                    alignment_sigma_hours,
                )
                correction_l2 = components["image_correction"].square().mean()
                loss = (
                    forecast_loss
                    + wind_aux_weight * wind_loss
                    + alignment_weight * alignment_value
                    + correction_l2_weight * correction_l2
                )

            if training:
                scaler.scale(loss).backward()
                if gradient_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip
                    )
                scaler.step(optimizer)
                scaler.update()

        forecast_error = (prediction.detach().float() - target.float()) * 1000.0
        wind_error = (
            components["wind_base"].detach().float() - target.float()
        ) * 1000.0
        ar_error = (
            components["ar_base"].detach().float() - target.float()
        ) * 1000.0
        correction = components["image_correction"].detach().float() * 1000.0
        totals["forecast"] += float(forecast_error.square().sum().cpu())
        totals["wind_base"] += float(wind_error.square().sum().cpu())
        totals["ar_base"] += float(ar_error.square().sum().cpu())
        totals["correction"] += float(correction.square().sum().cpu())
        totals["alignment"] += float(alignment_value.detach().cpu()) * len(images)
        value_count += forecast_error.numel()
        sample_count += len(images)

        batch_chain_ids = batch["chain_id"].numpy()
        row_squared_error = forecast_error.square().sum(dim=1).cpu().numpy()
        np.add.at(chain_squared_error, batch_chain_ids, row_squared_error)
        np.add.at(chain_value_count, batch_chain_ids, forecast_error.shape[1])

        for name, value in model.training_diagnostics().items():
            scalar = float(torch.as_tensor(value).detach().float().cpu())
            diagnostics_sum[name] = (
                diagnostics_sum.get(name, 0.0) + scalar * len(images)
            )

        if collect_predictions:
            predictions.append(prediction.detach().float().cpu().numpy() * 1000.0)
            sample_ids.extend(batch["sample_id"])
            chain_ids_output.extend(batch_chain_ids.tolist())
            chain_positions_output.extend(
                batch["chain_position"].numpy().tolist()
            )

        if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(loader):
            mode = "train" if training else "val"
            print(
                f"{mode} batch={batch_index}/{len(loader)} "
                f"running_rmse={math.sqrt(totals['forecast'] / value_count):.3f} "
                f"wind_rmse={math.sqrt(totals['wind_base'] / value_count):.3f} "
                f"ar_rmse={math.sqrt(totals['ar_base'] / value_count):.3f} "
                f"corr_rms={math.sqrt(totals['correction'] / value_count):.3f} "
                f"lag_kl={totals['alignment'] / sample_count:.3f}",
                flush=True,
            )

    valid_chains = chain_value_count > 0
    result = {
        "rmse": math.sqrt(totals["forecast"] / value_count),
        "chain_macro_rmse": float(
            np.sqrt(
                chain_squared_error[valid_chains]
                / chain_value_count[valid_chains]
            ).mean()
        ),
        "wind_base_rmse": math.sqrt(totals["wind_base"] / value_count),
        "ar_base_rmse": math.sqrt(totals["ar_base"] / value_count),
        "correction_rms": math.sqrt(totals["correction"] / value_count),
        "alignment_kl": totals["alignment"] / sample_count,
        "diagnostics": {
            name: value / sample_count for name, value in diagnostics_sum.items()
        },
    }
    if collect_predictions:
        result.update(
            {
                "predictions": np.concatenate(predictions),
                "sample_ids": sample_ids,
                "chain_ids": np.asarray(chain_ids_output),
                "chain_positions": np.asarray(chain_positions_output),
            }
        )
    return result


def main():
    wind_only = os.getenv("WIND_ONLY", "0").lower() in {"1", "true", "yes"}
    chain_balanced = os.getenv("CHAIN_BALANCED_SAMPLING", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    train_chains = infer_temporal_chains(train_inputs, IMAGE_COLUMNS)
    val_chains = infer_temporal_chains(val_inputs, IMAGE_COLUMNS)
    ar_fit, ar_scale, ar_val_micro, ar_val_macro = fit_ar_configuration(
        train_chains, val_chains
    )

    selected_train_index = limited_indexes(train_index, "MAX_TRAIN_SAMPLES")
    selected_val_index = limited_indexes(val_index, "MAX_VAL_SAMPLES")
    train_dataset = ChainAwareSolarWindDataset(
        train_image_array,
        train_image_index,
        train_inputs,
        selected_train_index,
        train_targets,
        temporal_chains=train_chains,
    )
    val_dataset = ChainAwareSolarWindDataset(
        val_image_array,
        val_image_index,
        val_inputs,
        selected_val_index,
        val_targets,
        temporal_chains=val_chains,
    )
    train_loader = make_chain_loader(
        train_dataset, training=True, chain_balanced=chain_balanced
    )
    val_loader = make_chain_loader(val_dataset, training=False)

    model_kwargs = build_model_kwargs(ar_fit, ar_scale, wind_only=wind_only)
    model = MODEL_CLASS(**model_kwargs).to(DEVICE)
    peak_lr = float(os.getenv("LEARNING_RATE", "5e-5"))
    weight_decay = float(os.getenv("V12_WEIGHT_DECAY", "0.02"))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=peak_lr, weight_decay=weight_decay
    )
    scheduler, warmup_epochs, minimum_lr = make_scheduler(optimizer, peak_lr)
    scaler = torch.amp.GradScaler(AMP_DEVICE_TYPE, enabled=USE_AMP)

    wind_aux_weight = float(os.getenv("V12_WIND_AUX_WEIGHT", "0.20"))
    alignment_weight = float(os.getenv("V12_LAG_ALIGNMENT_WEIGHT", "0.01"))
    alignment_sigma_hours = float(
        os.getenv("V12_ALIGNMENT_SIGMA_HOURS", "12")
    )
    correction_l2_weight = float(
        os.getenv("V12_CORRECTION_L2_WEIGHT", "0.002")
    )
    gradient_clip = float(os.getenv("V12_GRADIENT_CLIP", "1.0"))
    patience = int(os.getenv("V12_EARLY_STOP_PATIENCE", "10"))

    manifest = {
        "train": chain_manifest(train_chains),
        "validation": chain_manifest(val_chains),
        "train_rows_used": int(len(selected_train_index)),
        "validation_rows_used": int(len(selected_val_index)),
        "source_models": [
            "competition CNN-LSTM baseline",
            "Seokho source-map and speed-dependent lag findings",
        ],
        MANIFEST_VERSION_KEY: MODEL_CHANGES,
    }
    (OUTPUT_DIR / f"{FILE_STEM}_chain_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    checkpoint_path = OUTPUT_DIR / f"best_{FILE_STEM}.pth"
    history_path = OUTPUT_DIR / f"{FILE_STEM}_history.csv"
    validation_path = OUTPUT_DIR / f"{FILE_STEM}_validation_predictions.csv"
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(
        f"architecture={FILE_STEM} device={DEVICE} parameters={parameter_count:,} "
        f"wind_only={wind_only} image_size={IMAGE_SIZE} "
        f"spatial_grid={model.grid_rows}x{model.grid_columns} "
        f"frame_dim={model.frame_dim} lstm_hidden={model.lstm_hidden_dim} "
        f"lag_hours={model.lag_hours.tolist()} "
        f"lag_prior_max={model.lag_prior_max_strength:.2f} "
        f"train_chains={train_chains.count} val_chains={val_chains.count} "
        f"chain_balanced={chain_balanced} lr={peak_lr:.2e} "
        f"mask={SOLAR_DISK_MASK} norm={IMAGE_NORM} "
        f"loss=forecast_rmse+wind_aux+weak_lag_kl+correction_l2"
    )

    best_val_rmse = float("inf")
    epochs_without_improvement = 0
    history = []
    val_target_rows = val_targets[selected_val_index]
    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            train_chains.count,
            optimizer,
            scaler,
            True,
            wind_aux_weight,
            alignment_weight,
            alignment_sigma_hours,
            correction_l2_weight,
            gradient_clip,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                val_chains.count,
                optimizer,
                scaler,
                False,
                wind_aux_weight,
                alignment_weight,
                alignment_sigma_hours,
                correction_l2_weight,
                gradient_clip,
                collect_predictions=True,
            )
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "train_rmse_km_s": train_metrics["rmse"],
            "val_rmse_km_s": val_metrics["rmse"],
            "train_chain_macro_rmse_km_s": train_metrics[
                "chain_macro_rmse"
            ],
            "val_chain_macro_rmse_km_s": val_metrics["chain_macro_rmse"],
            "val_ar_base_rmse_km_s": val_metrics["ar_base_rmse"],
            "val_wind_base_rmse_km_s": val_metrics["wind_base_rmse"],
            "val_image_correction_rms_km_s": val_metrics["correction_rms"],
            "train_lag_alignment_kl": train_metrics["alignment_kl"],
            "val_lag_alignment_kl": val_metrics["alignment_kl"],
            "learning_rate": learning_rate,
            "seconds": elapsed,
            **{
                f"val_{name}": value
                for name, value in val_metrics["diagnostics"].items()
            },
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        diagnostics = val_metrics["diagnostics"]
        print(
            f"epoch={epoch:03d} train_rmse={train_metrics['rmse']:.3f} "
            f"val_rmse={val_metrics['rmse']:.3f} "
            f"val_chain_macro_rmse={val_metrics['chain_macro_rmse']:.3f} "
            f"wind_val_rmse={val_metrics['wind_base_rmse']:.3f} "
            f"corr_rms={val_metrics['correction_rms']:.3f} "
            f"lag_kl={val_metrics['alignment_kl']:.3f} "
            f"attention_delay_h={diagnostics['attention_delay_h']:.1f} "
            f"attention_entropy={diagnostics['attention_entropy']:.3f} "
            f"expected_lag_h={diagnostics['expected_lag_h']:.1f} "
            f"gate={diagnostics['correction_gate']:.3f} "
            f"lr={learning_rate:.2e} seconds={elapsed:.1f}",
            flush=True,
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "architecture": ARCHITECTURE_NAME,
                    "version": CHECKPOINT_VERSION,
                    "model_state_dict": model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "epoch": epoch,
                    "val_rmse_km_s": val_metrics["rmse"],
                    "val_chain_macro_rmse_km_s": val_metrics[
                        "chain_macro_rmse"
                    ],
                    "channels": CHANNELS,
                    "ar_validation": {
                        "micro_rmse_km_s": ar_val_micro,
                        "chain_macro_rmse_km_s": ar_val_macro,
                    },
                    "preprocess": {
                        "image_size": IMAGE_SIZE,
                        "image_norm": IMAGE_NORM,
                        "soft_cubic_strength": SOFT_CUBIC_STRENGTH,
                        "solar_disk_mask": SOLAR_DISK_MASK,
                        "solar_disk_radius_fraction": SOLAR_DISK_RADIUS_FRACTION,
                        "feature_schema": FEATURE_SCHEMA,
                        "spatial_grid": [model.grid_rows, model.grid_columns],
                        "lag_hours": model.lag_hours.tolist(),
                        "lag_prior_max_strength": model.lag_prior_max_strength,
                        "lag_alignment_weight": alignment_weight,
                        "warmup_epochs": warmup_epochs,
                        "minimum_learning_rate": minimum_lr,
                        "optimizer_weight_decay": weight_decay,
                        **EXTRA_PREPROCESS,
                    },
                },
                checkpoint_path,
            )
            save_validation_predictions(
                val_metrics, val_target_rows, validation_path
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("early stopping")
                break

    history_frame = pd.DataFrame(history)
    history_frame.plot(
        x="epoch", y=["train_rmse_km_s", "val_rmse_km_s"], grid=True
    )
    plt.ylabel("RMSE (km/s)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{FILE_STEM}_learning_curve.png", dpi=140)
    plt.close()
    print(
        f"saved: {checkpoint_path.resolve()} best_val_rmse={best_val_rmse:.3f}"
    )


if __name__ == "__main__":
    main()
