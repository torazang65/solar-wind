import json
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

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
from model_solar_timing_transformer_v13 import (
    ARCHITECTURE_NAME,
    FILE_STEM,
    FORECAST_STEPS,
    HINDCAST_STEPS,
    QUERY_STEPS,
    SolarWindTimingTransformerV13,
)
from train_solar_hybrid_v10 import limited_indexes, save_validation_predictions
from train_solar_lstm_v12 import fit_ar_configuration


FEATURE_SCHEMA = "disk_mask_signed_delta_lite_unet_speed_locked_timing_v13"
CHECKPOINT_VERSION = "13"
HOURS_AT_1_AU_PER_1000_KMS = 149_597_870.7 / 1000.0 / 3600.0
MODEL_CLASS = SolarWindTimingTransformerV13
SOURCE_MODELS = [
    "competition CNN-LSTM sequence pipeline",
    "Seokho V7 source-speed-to-arrival mapping",
    "Taeuk V12.1 Lite U-Net spatial encoder",
]
MODEL_CHANGES = [
    "image-only source-speed map; no neural wind shortcut",
    "source speed and transit timing share the same scalar",
    "physics-biased Transformer over 13 hindcast and 12 forecast queries",
    "strict acquisition-time causal mask for every query",
    "train-only AR(2) fallback with at most 0.5 image blend",
    "weak target-derived backmapping alignment ablation",
]
MANIFEST_VERSION_KEY = "v13_changes"
EXTRA_PREPROCESS = {}
AUXILIARY_OBJECTIVE = None


def boolean_environment(name, default=False):
    fallback = "1" if default else "0"
    return os.getenv(name, fallback).lower() in {"1", "true", "yes"}


def parse_unet_channels(value=None):
    text = value if value is not None else os.getenv(
        "V13_UNET_CHANNELS", "12,16,24,40,56"
    )
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if len(values) != 5 or any(value <= 0 for value in values):
        raise ValueError("V13_UNET_CHANNELS must contain five positive integers")
    return values


def build_model_kwargs(ar_fit, ar_residual_scale, wind_only=False):
    prior_min = float(os.getenv("V13_PHYSICAL_PRIOR_MIN", "1.0"))
    prior_max = float(os.getenv("V13_PHYSICAL_PRIOR_MAX", "4.0"))
    prior_init = float(os.getenv("V13_PHYSICAL_PRIOR_INIT", "2.0"))
    return {
        "image_size": IMAGE_SIZE,
        "use_images": not wind_only,
        "ar_coefficients": ar_fit.coefficients.tolist(),
        "ar_intercept": ar_fit.intercept,
        "baseline_residual_scale": ar_residual_scale.tolist(),
        "grid_rows": int(os.getenv("V13_GRID_ROWS", "2")),
        "grid_columns": int(os.getenv("V13_GRID_COLUMNS", "8")),
        "unet_channels": list(parse_unet_channels()),
        "d_model": int(os.getenv("V13_D_MODEL", "96")),
        "attention_heads": int(os.getenv("V13_ATTENTION_HEADS", "4")),
        "decoder_layers": int(os.getenv("V13_DECODER_LAYERS", "2")),
        "feedforward_dim": int(os.getenv("V13_FEEDFORWARD_DIM", "192")),
        "dropout": float(os.getenv("V13_DROPOUT", "0.15")),
        "time_mask_prob": float(os.getenv("V13_TIME_MASK_PROBABILITY", "0.15")),
        "modality_drop_prob": float(
            os.getenv("V13_MODALITY_DROP_PROBABILITY", "0.25")
        ),
        "delta_gain": float(os.getenv("V13_DELTA_GAIN", "1.0")),
        "timing_sigma_hours": float(os.getenv("V13_TIMING_SIGMA_HOURS", "18")),
        "physical_prior_min_strength": prior_min,
        "physical_prior_max_strength": prior_max,
        "physical_prior_init_strength": prior_init,
        "maximum_blend": float(os.getenv("V13_MAXIMUM_BLEND", "0.50")),
        "initial_blend": float(os.getenv("V13_INITIAL_BLEND", "0.05")),
        "correction_cap_multiplier": float(
            os.getenv("V13_CORRECTION_CAP_MULTIPLIER", "1.0")
        ),
        "apply_solar_disk_mask": SOLAR_DISK_MASK,
        "solar_disk_center_fraction": SOLAR_DISK_CENTER_FRACTION,
        "solar_disk_radius_fraction": SOLAR_DISK_RADIUS_FRACTION,
        "solar_disk_edge_pixels": float(
            os.getenv("V13_SOLAR_DISK_EDGE_PIXELS", "1.5")
        ),
    }


def make_scheduler(optimizer, peak_lr):
    warmup_epochs = int(os.getenv("V13_WARMUP_EPOCHS", "3"))
    minimum_lr = float(os.getenv("V13_MIN_LR", "1e-6"))
    if EPOCHS == 1:
        return (
            torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
            0,
            minimum_lr,
        )
    if not 0 < warmup_epochs < EPOCHS:
        raise ValueError("V13_WARMUP_EPOCHS must be between 1 and EPOCHS - 1")

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


def backmapping_alignment_kl(model, aux, wind, target, sigma_deg):
    """Match timing attention to target-derived source longitude weakly."""
    with torch.amp.autocast(AMP_DEVICE_TYPE, enabled=False):
        attention = aux["timing_attention"].float().view(
            -1,
            QUERY_STEPS,
            20,
            model.grid_rows,
            model.grid_columns,
        )
        batch_size = attention.shape[0]
        query_hours = model.query_hours.float()
        y_series = torch.cat([wind[:, 7:], target], dim=1).float()
        transit_hours = (
            model.effective_distance().detach().float()
            / y_series.clamp(0.25, 0.90)
        )
        delta_time = (
            query_hours.view(1, QUERY_STEPS, 1)
            - transit_hours.unsqueeze(-1)
            + model.image_age_hours.float().view(1, 1, 20)
        )
        target_longitude = -model.omega_deg_per_hour * delta_time
        visible = target_longitude.abs() <= 90.0
        longitude_error = (
            model.cell_longitude_deg.float().view(
                1, 1, 1, model.grid_rows, model.grid_columns
            )
            - target_longitude.unsqueeze(-1).unsqueeze(-1)
        )
        distribution = torch.exp(
            -longitude_error.square() / (2.0 * float(sigma_deg) ** 2)
        )
        distribution = distribution * visible.unsqueeze(-1).unsqueeze(-1)
        distribution = distribution * aux["valid_source"].float().view_as(
            distribution
        )
        mass = distribution.sum(dim=(2, 3, 4))
        valid = (mass > 1e-6).float() * aux["image_keep"].float().unsqueeze(-1)
        distribution = distribution / mass.clamp_min(1e-6).view(
            batch_size, QUERY_STEPS, 1, 1, 1
        )
        learned = attention.clamp_min(1e-8)
        kl = (
            distribution
            * (distribution.clamp_min(1e-12).log() - learned.log())
        ).sum(dim=(2, 3, 4))
        return (kl * valid).sum() / valid.sum().clamp_min(1.0)


def source_speed_smoothness(aux):
    speed = aux["source_speed"]
    pair_keep = (
        aux["time_keep"][:, 1:] * aux["time_keep"][:, :-1]
    ).view(speed.shape[0], speed.shape[1] - 1, 1, 1)
    pair_keep = pair_keep * aux["image_keep"].view(-1, 1, 1, 1)
    squared_difference = (speed[:, 1:] - speed[:, :-1]).square()
    count = pair_keep.sum() * speed.shape[2] * speed.shape[3]
    return (squared_difference * pair_keep).sum() / count.clamp_min(1.0)


def run_epoch(
    model,
    loader,
    chain_count,
    optimizer,
    scaler,
    training,
    hindcast_weight,
    alignment_weight,
    alignment_sigma_deg,
    correction_l2_weight,
    gate_l1_weight,
    speed_smoothness_weight,
    gradient_clip,
    collect_predictions=False,
):
    model.train(training)
    totals = {
        "forecast": 0.0,
        "ar_base": 0.0,
        "correction": 0.0,
        "hindcast": 0.0,
        "alignment": 0.0,
    }
    counts = {"forecast": 0, "hindcast": 0, "alignment": 0, "samples": 0}
    chain_squared_error = np.zeros(chain_count, dtype=np.float64)
    chain_value_count = np.zeros(chain_count, dtype=np.int64)
    diagnostics_sum = {}
    auxiliary_sum = {}
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
                    images, wind, return_components=True, return_aux=True
                )
                forecast_loss = torch.sqrt(
                    F.mse_loss(prediction, target) + RMSE_EPSILON
                )
                keep = aux["image_keep"].unsqueeze(-1)
                hindcast_target = wind[:, 7:]
                hindcast_count = (
                    keep.sum() * HINDCAST_STEPS
                ).clamp_min(1.0)
                hindcast_mse = (
                    (aux["hindcast"] - hindcast_target).square() * keep
                ).sum() / hindcast_count
                hindcast_loss = torch.sqrt(hindcast_mse + RMSE_EPSILON)
                alignment_value = backmapping_alignment_kl(
                    model, aux, wind, target, alignment_sigma_deg
                )
                correction_l2 = components["image_correction"].square().mean()
                gate_l1 = components["correction_gate"].mean()
                smoothness = source_speed_smoothness(aux)
                auxiliary_loss = prediction.new_zeros(())
                auxiliary_metrics = {}
                if AUXILIARY_OBJECTIVE is not None:
                    auxiliary_loss, auxiliary_metrics = AUXILIARY_OBJECTIVE(
                        model, prediction, components, aux, wind, target
                    )
                loss = (
                    forecast_loss
                    + hindcast_weight * hindcast_loss
                    + alignment_weight * alignment_value
                    + correction_l2_weight * correction_l2
                    + gate_l1_weight * gate_l1
                    + speed_smoothness_weight * smoothness
                    + auxiliary_loss
                )

            if training:
                scaler.scale(loss).backward()
                if gradient_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()

        forecast_error = (prediction.detach().float() - target.float()) * 1000.0
        ar_error = (components["ar_base"].detach().float() - target.float()) * 1000.0
        correction = components["image_correction"].detach().float() * 1000.0
        hindcast_error = (
            aux["hindcast"].detach().float() - hindcast_target.float()
        ) * 1000.0
        totals["forecast"] += float(forecast_error.square().sum().cpu())
        totals["ar_base"] += float(ar_error.square().sum().cpu())
        totals["correction"] += float(correction.square().sum().cpu())
        totals["hindcast"] += float(
            (hindcast_error.square() * keep).sum().cpu()
        )
        totals["alignment"] += float(alignment_value.detach().cpu()) * len(images)
        counts["forecast"] += forecast_error.numel()
        counts["hindcast"] += int((keep.sum() * HINDCAST_STEPS).cpu())
        counts["alignment"] += len(images)
        counts["samples"] += len(images)
        for name, value in auxiliary_metrics.items():
            scalar = float(torch.as_tensor(value).detach().float().cpu())
            auxiliary_sum[name] = (
                auxiliary_sum.get(name, 0.0) + scalar * len(images)
            )

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
            chain_positions_output.extend(batch["chain_position"].numpy().tolist())

        if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(loader):
            mode = "train" if training else "val"
            hindcast_rmse = (
                math.sqrt(totals["hindcast"] / counts["hindcast"])
                if counts["hindcast"] > 0
                else float("nan")
            )
            print(
                f"{mode} batch={batch_index}/{len(loader)} "
                f"running_rmse={math.sqrt(totals['forecast'] / counts['forecast']):.3f} "
                f"ar_rmse={math.sqrt(totals['ar_base'] / counts['forecast']):.3f} "
                f"corr_rms={math.sqrt(totals['correction'] / counts['forecast']):.3f} "
                f"hind_rmse={hindcast_rmse:.3f} "
                f"align_kl={totals['alignment'] / counts['alignment']:.3f}",
                flush=True,
            )

    valid_chains = chain_value_count > 0
    result = {
        "rmse": math.sqrt(totals["forecast"] / counts["forecast"]),
        "chain_macro_rmse": float(
            np.sqrt(
                chain_squared_error[valid_chains] / chain_value_count[valid_chains]
            ).mean()
        ),
        "ar_base_rmse": math.sqrt(totals["ar_base"] / counts["forecast"]),
        "correction_rms": math.sqrt(totals["correction"] / counts["forecast"]),
        "hindcast_rmse": (
            math.sqrt(totals["hindcast"] / counts["hindcast"])
            if counts["hindcast"] > 0
            else float("nan")
        ),
        "alignment_kl": totals["alignment"] / counts["alignment"],
        "diagnostics": {
            name: value / counts["samples"]
            for name, value in diagnostics_sum.items()
        },
        "auxiliary_metrics": {
            name: value / counts["samples"]
            for name, value in auxiliary_sum.items()
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
    wind_only = boolean_environment("WIND_ONLY")
    chain_balanced = boolean_environment("CHAIN_BALANCED_SAMPLING")
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
    peak_lr = float(os.getenv("LEARNING_RATE", "3e-5"))
    weight_decay = float(os.getenv("V13_WEIGHT_DECAY", "0.03"))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=peak_lr, weight_decay=weight_decay
    )
    scheduler, warmup_epochs, minimum_lr = make_scheduler(optimizer, peak_lr)
    scaler = torch.amp.GradScaler(AMP_DEVICE_TYPE, enabled=USE_AMP)

    hindcast_weight_start = float(os.getenv("V13_HINDCAST_WEIGHT_START", "0.50"))
    hindcast_weight_end = float(os.getenv("V13_HINDCAST_WEIGHT_END", "0.10"))
    hindcast_decay_epochs = int(os.getenv("V13_HINDCAST_DECAY_EPOCHS", "8"))
    alignment_weight = float(os.getenv("V13_ALIGNMENT_WEIGHT", "0.01"))
    alignment_sigma_deg = float(os.getenv("V13_ALIGNMENT_SIGMA_DEG", "12"))
    correction_l2_weight = float(os.getenv("V13_CORRECTION_L2_WEIGHT", "0.10"))
    gate_l1_weight = float(os.getenv("V13_GATE_L1_WEIGHT", "0.01"))
    speed_smoothness_weight = float(
        os.getenv("V13_SPEED_SMOOTHNESS_WEIGHT", "0.02")
    )
    gradient_clip = float(os.getenv("V13_GRADIENT_CLIP", "1.0"))
    patience = int(os.getenv("V13_EARLY_STOP_PATIENCE", "6"))

    manifest = {
        "train": chain_manifest(train_chains),
        "validation": chain_manifest(val_chains),
        "train_rows_used": int(len(selected_train_index)),
        "validation_rows_used": int(len(selected_val_index)),
        "source_models": SOURCE_MODELS,
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
        f"source_grid=20x{model.grid_rows}x{model.grid_columns} "
        f"d_model={model.d_model} heads={model.attention_heads} "
        f"layers={model.decoder_layers} train_chains={train_chains.count} "
        f"val_chains={val_chains.count} chain_balanced={chain_balanced} "
        f"lr={peak_lr:.2e} mask={SOLAR_DISK_MASK} norm={IMAGE_NORM} "
        f"physical_prior=[{model.physical_prior_min_strength:.1f},"
        f"{model.physical_prior_max_strength:.1f}] "
        f"max_blend={model.maximum_blend:.2f} "
        f"loss=forecast+decayed_hindcast+optional_backmapping+guardrails"
    )

    best_val_rmse = float("inf")
    epochs_without_improvement = 0
    history = []
    val_target_rows = val_targets[selected_val_index]
    for epoch in range(1, EPOCHS + 1):
        decay_progress = min(1.0, (epoch - 1) / max(1, hindcast_decay_epochs - 1))
        hindcast_weight = (
            hindcast_weight_start
            + decay_progress * (hindcast_weight_end - hindcast_weight_start)
        )
        started = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            train_chains.count,
            optimizer,
            scaler,
            True,
            hindcast_weight,
            alignment_weight,
            alignment_sigma_deg,
            correction_l2_weight,
            gate_l1_weight,
            speed_smoothness_weight,
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
                hindcast_weight,
                alignment_weight,
                alignment_sigma_deg,
                correction_l2_weight,
                gate_l1_weight,
                speed_smoothness_weight,
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
            "train_chain_macro_rmse_km_s": train_metrics["chain_macro_rmse"],
            "val_chain_macro_rmse_km_s": val_metrics["chain_macro_rmse"],
            "val_ar_base_rmse_km_s": val_metrics["ar_base_rmse"],
            "val_image_correction_rms_km_s": val_metrics["correction_rms"],
            "train_hindcast_rmse_km_s": train_metrics["hindcast_rmse"],
            "val_hindcast_rmse_km_s": val_metrics["hindcast_rmse"],
            "train_backmapping_kl": train_metrics["alignment_kl"],
            "val_backmapping_kl": val_metrics["alignment_kl"],
            "hindcast_weight": hindcast_weight,
            "learning_rate": learning_rate,
            "seconds": elapsed,
            **{
                f"val_{name}": value
                for name, value in val_metrics["diagnostics"].items()
            },
            **{
                f"train_{name}": value
                for name, value in train_metrics["auxiliary_metrics"].items()
            },
            **{
                f"val_{name}": value
                for name, value in val_metrics["auxiliary_metrics"].items()
            },
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        diagnostics = val_metrics["diagnostics"]
        auxiliary_text = "".join(
            f" {name}={value:.3f}"
            for name, value in val_metrics["auxiliary_metrics"].items()
        )
        print(
            f"epoch={epoch:03d} train_rmse={train_metrics['rmse']:.3f} "
            f"val_rmse={val_metrics['rmse']:.3f} "
            f"val_chain_macro_rmse={val_metrics['chain_macro_rmse']:.3f} "
            f"ar_val_rmse={val_metrics['ar_base_rmse']:.3f} "
            f"corr_rms={val_metrics['correction_rms']:.3f} "
            f"hind={val_metrics['hindcast_rmse']:.1f} "
            f"align_kl={val_metrics['alignment_kl']:.3f} "
            f"delay_h={diagnostics['attention_delay_h']:.1f} "
            f"entropy={diagnostics['attention_entropy']:.3f} "
            f"speed={diagnostics['source_speed_mean_kms']:.1f}+/-"
            f"{diagnostics['source_speed_std_kms']:.1f} "
            f"distance={diagnostics['effective_distance_h']:.2f} "
            f"prior={diagnostics['physical_prior_strength']:.2f} "
            f"gate={diagnostics['correction_gate']:.3f} "
            f"lr={learning_rate:.2e} seconds={elapsed:.1f}"
            f"{auxiliary_text}",
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
                        "source_grid": [model.grid_rows, model.grid_columns],
                        "unet_channels": list(model.unet_channels),
                        "timing_sigma_hours": model.timing_sigma_hours,
                        "physical_prior_bounds": [
                            model.physical_prior_min_strength,
                            model.physical_prior_max_strength,
                        ],
                        "maximum_blend": model.maximum_blend,
                        "alignment_weight": alignment_weight,
                        "alignment_sigma_deg": alignment_sigma_deg,
                        "hindcast_weight_start": hindcast_weight_start,
                        "hindcast_weight_end": hindcast_weight_end,
                        "hindcast_decay_epochs": hindcast_decay_epochs,
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
