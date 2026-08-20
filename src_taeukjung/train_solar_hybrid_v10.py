import copy
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
from model_solar_hybrid_v10 import (
    ARCHITECTURE_NAME,
    FILE_STEM,
    SolarWindAnchoredHybridV10,
)


class ExponentialMovingAverage:
    def __init__(self, model, decay):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be between zero and one")
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)
        self.decay = float(decay)
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        source_state = model.state_dict()
        for name, averaged in self.model.state_dict().items():
            source = source_state[name].detach()
            if averaged.is_floating_point():
                averaged.lerp_(source, 1.0 - decay)
            else:
                averaged.copy_(source)


def limited_indexes(indexes, environment_name):
    limit = int(os.getenv(environment_name, "0"))
    if limit <= 0 or limit >= len(indexes):
        return indexes
    return indexes[:limit]


def fit_ar_configuration(train_chains, val_chains):
    order = int(os.getenv("V10_AR_ORDER", "2"))
    ridge_strength = float(os.getenv("V10_AR_RIDGE", "30"))
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
    return {
        "image_size": IMAGE_SIZE,
        "use_images": not wind_only,
        "ar_coefficients": ar_fit.coefficients.tolist(),
        "ar_intercept": ar_fit.intercept,
        "ar_ridge_strength": ar_fit.ridge_strength,
        "baseline_residual_scale": ar_residual_scale.tolist(),
        "d_model": int(os.getenv("V10_D_MODEL", "128")),
        "nhead": int(os.getenv("V10_NHEAD", "8")),
        "num_encoder_layers": int(os.getenv("V10_ENCODER_LAYERS", "2")),
        "num_decoder_layers": int(os.getenv("V10_DECODER_LAYERS", "1")),
        "dim_feedforward": int(os.getenv("V10_FF_DIM", "256")),
        "dropout": float(os.getenv("V10_DROPOUT", "0.15")),
        "time_mask_prob": float(os.getenv("V10_TIME_MASK_PROBABILITY", "0.05")),
        "modality_drop_prob": float(
            os.getenv("V10_MODALITY_DROP_PROBABILITY", "0.20")
        ),
        "use_surge_head": True,
        "wind_feature_dim": int(os.getenv("V10_WIND_FEATURE_DIM", "64")),
        "wind_residual_cap_multiplier": float(
            os.getenv("V10_WIND_RESIDUAL_CAP_MULTIPLIER", "1.0")
        ),
        "propagation_cap_multiplier": float(
            os.getenv("V10_PROPAGATION_CAP_MULTIPLIER", "1.25")
        ),
        "correction_cap_multiplier": float(
            os.getenv("V10_CORRECTION_CAP_MULTIPLIER", "0.75")
        ),
        "correction_drop_prob": float(
            os.getenv("V10_CORRECTION_DROP_PROBABILITY", "0.30")
        ),
        "fixed_lag_hours": float(os.getenv("V10_FIXED_LAG_HOURS", "96")),
        "fixed_lag_reference_speed_kms": float(
            os.getenv("V10_FIXED_LAG_REFERENCE_SPEED_KMS", "430")
        ),
        "delta_gain": float(os.getenv("V10_DELTA_GAIN", "4.0")),
        "apply_solar_disk_mask": SOLAR_DISK_MASK,
        "solar_disk_center_fraction": SOLAR_DISK_CENTER_FRACTION,
        "solar_disk_radius_fraction": SOLAR_DISK_RADIUS_FRACTION,
        "solar_disk_edge_pixels": float(
            os.getenv("V10_SOLAR_DISK_EDGE_PIXELS", "1.5")
        ),
        "climatology_speed_kms": float(
            os.getenv("V10_CLIMATOLOGY_SPEED_KMS", "430")
        ),
    }


def is_physical_parameter(name):
    return (
        name == "fallback_weight_raw"
        or name.startswith("fusion_gate_head.module.")
        or name in {
            "source_speed_head.bias",
            "source_gate_head.bias",
            "transit_residual_head.bias",
        }
    )


def make_optimizer(model):
    peak_lr = float(os.getenv("LEARNING_RATE", "5e-5"))
    weight_decay = float(os.getenv("V10_WEIGHT_DECAY", "0.02"))
    physical_multiplier = float(os.getenv("V10_PHYSICAL_LR_MULT", "20"))
    base_parameters = []
    physical_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = physical_parameters if is_physical_parameter(name) else base_parameters
        target.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": base_parameters,
                "lr": peak_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": physical_parameters,
                "lr": peak_lr * physical_multiplier,
                "weight_decay": 0.0,
            },
        ]
    )
    return optimizer, peak_lr, weight_decay, physical_multiplier


def make_scheduler(optimizer, peak_lr):
    warmup_epochs = int(os.getenv("V10_WARMUP_EPOCHS", "3"))
    minimum_lr = float(os.getenv("V10_MIN_LR", "1e-6"))
    if not 0 < warmup_epochs < EPOCHS:
        raise ValueError("V10_WARMUP_EPOCHS must be between 1 and EPOCHS - 1")

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


def rank_auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = ranks[labels].sum()
    return float(
        (rank_sum - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def run_epoch(
    model,
    loader,
    chain_count,
    optimizer,
    scaler,
    training,
    hindcast_weight,
    wind_aux_weight,
    transit_l2_weight,
    surge_weight,
    surge_pos_weight,
    component_l2_weight,
    ema=None,
    collect_predictions=False,
):
    model.train(training)
    totals = {
        "forecast": 0.0,
        "wind": 0.0,
        "ar": 0.0,
        "wind_residual": 0.0,
        "propagation": 0.0,
        "correction": 0.0,
        "hindcast": 0.0,
    }
    counts = {"forecast": 0, "hindcast": 0}
    chain_squared_error = np.zeros(chain_count, dtype=np.float64)
    chain_value_count = np.zeros(chain_count, dtype=np.int64)
    diagnostics_sum = {}
    diagnostics_count = 0
    predictions = []
    sample_ids = []
    chain_ids_output = []
    chain_positions_output = []
    surge_scores = []
    surge_labels = []

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
                forecast_rmse = torch.sqrt(
                    F.mse_loss(prediction, target) + RMSE_EPSILON
                )
                wind_rmse = torch.sqrt(
                    F.mse_loss(components["wind_prediction"], target)
                    + RMSE_EPSILON
                )
                loss = forecast_rmse + wind_aux_weight * wind_rmse

                if aux["hindcast"] is not None:
                    keep = aux["image_keep"].unsqueeze(-1)
                    hindcast_target = wind[:, 7:]
                    kept_values = (
                        keep.sum() * hindcast_target.shape[1]
                    ).clamp_min(1.0)
                    hindcast_mse = (
                        (aux["hindcast"] - hindcast_target).square() * keep
                    ).sum() / kept_values
                    loss = loss + hindcast_weight * torch.sqrt(
                        hindcast_mse + RMSE_EPSILON
                    )
                    loss = loss + transit_l2_weight * aux[
                        "transit_residual"
                    ].square().mean()

                surge_label = (
                    (target.max(dim=1).values - wind[:, -1])
                    > float(os.getenv("V10_SURGE_THRESHOLD_KMS", "100")) / 1000.0
                ).float().unsqueeze(-1)
                if aux["surge_logit"] is not None:
                    keep = aux["image_keep"].unsqueeze(-1)
                    surge_loss = F.binary_cross_entropy_with_logits(
                        aux["surge_logit"],
                        surge_label,
                        weight=keep,
                        pos_weight=surge_pos_weight,
                        reduction="sum",
                    ) / keep.sum().clamp_min(1.0)
                    loss = loss + surge_weight * surge_loss

                if component_l2_weight > 0.0:
                    scale = model.hybrid_residual_scale.to(
                        dtype=prediction.dtype
                    )
                    standardized_energy = sum(
                        (components[name] / scale).square().mean()
                        for name in ("wind_residual", "propagation_residual", "correction")
                    )
                    loss = loss + component_l2_weight * standardized_energy

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)

        errors = {
            "forecast": (prediction.detach().float() - target.float()) * 1000.0,
            "wind": (
                components["wind_prediction"].detach().float() - target.float()
            )
            * 1000.0,
            "ar": (components["ar_baseline"].detach().float() - target.float())
            * 1000.0,
        }
        values = {
            "wind_residual": components["wind_residual"].detach().float() * 1000.0,
            "propagation": components["propagation_residual"].detach().float()
            * 1000.0,
            "correction": components["correction"].detach().float() * 1000.0,
        }
        for name, value in errors.items():
            totals[name] += float(value.square().sum().cpu())
        for name, value in values.items():
            totals[name] += float(value.square().sum().cpu())
        counts["forecast"] += errors["forecast"].numel()

        if aux["hindcast"] is not None:
            keep = aux["image_keep"].unsqueeze(-1)
            hindcast_error = (aux["hindcast"].detach() - wind[:, 7:]) * 1000.0
            totals["hindcast"] += float((hindcast_error.square() * keep).sum().cpu())
            counts["hindcast"] += int((keep.sum() * hindcast_error.shape[1]).cpu())

        batch_chain_ids = batch["chain_id"].numpy()
        row_squared_error = errors["forecast"].square().sum(dim=1).cpu().numpy()
        np.add.at(chain_squared_error, batch_chain_ids, row_squared_error)
        np.add.at(chain_value_count, batch_chain_ids, errors["forecast"].shape[1])

        for name, value in model.training_diagnostics().items():
            scalar = float(torch.as_tensor(value).detach().float().cpu())
            diagnostics_sum[name] = (
                diagnostics_sum.get(name, 0.0) + scalar * images.shape[0]
            )
        diagnostics_count += images.shape[0]

        if not training and aux["surge_logit"] is not None:
            surge_scores.extend(
                torch.sigmoid(aux["surge_logit"].detach())
                .squeeze(-1)
                .cpu()
                .numpy()
                .tolist()
            )
            surge_labels.extend(surge_label.squeeze(-1).cpu().numpy().tolist())

        if collect_predictions:
            predictions.append(prediction.detach().float().cpu().numpy() * 1000.0)
            sample_ids.extend(batch["sample_id"])
            chain_ids_output.extend(batch_chain_ids.tolist())
            chain_positions_output.extend(batch["chain_position"].numpy().tolist())

        if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(loader):
            mode = "train" if training else "val"
            print(
                f"{mode} batch={batch_index}/{len(loader)} "
                f"running_rmse={math.sqrt(totals['forecast'] / counts['forecast']):.3f} "
                f"wind_rmse={math.sqrt(totals['wind'] / counts['forecast']):.3f} "
                f"prop_rms={math.sqrt(totals['propagation'] / counts['forecast']):.3f} "
                f"corr_rms={math.sqrt(totals['correction'] / counts['forecast']):.3f}",
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
        "wind_rmse": math.sqrt(totals["wind"] / counts["forecast"]),
        "ar_rmse": math.sqrt(totals["ar"] / counts["forecast"]),
        "wind_residual_rms": math.sqrt(
            totals["wind_residual"] / counts["forecast"]
        ),
        "propagation_rms": math.sqrt(totals["propagation"] / counts["forecast"]),
        "correction_rms": math.sqrt(totals["correction"] / counts["forecast"]),
        "hindcast_rmse": (
            math.sqrt(totals["hindcast"] / counts["hindcast"])
            if counts["hindcast"] > 0
            else float("nan")
        ),
        "surge_auroc": rank_auc(surge_scores, surge_labels),
        "diagnostics": {
            name: value / diagnostics_count
            for name, value in diagnostics_sum.items()
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


def save_validation_predictions(metrics, targets, path):
    predictions = pd.DataFrame(
        metrics["predictions"],
        columns=[f"prediction_{column}" for column in TARGET_COLUMNS],
    )
    predictions.insert(0, "chain_position", metrics["chain_positions"])
    predictions.insert(0, "chain_id", metrics["chain_ids"])
    predictions.insert(0, "sample_id", metrics["sample_ids"])
    actual = pd.DataFrame(
        targets, columns=[f"actual_{column}" for column in TARGET_COLUMNS]
    )
    pd.concat([predictions, actual], axis=1).to_csv(path, index=False)


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
        north_south_flip_probability=0.0,
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
    model = SolarWindAnchoredHybridV10(**model_kwargs).to(DEVICE)
    optimizer, peak_lr, weight_decay, physical_multiplier = make_optimizer(model)
    scheduler, warmup_epochs, minimum_lr = make_scheduler(optimizer, peak_lr)
    scaler = torch.amp.GradScaler(AMP_DEVICE_TYPE, enabled=USE_AMP)
    ema_decay = float(os.getenv("V10_EMA_DECAY", "0.995"))
    ema = ExponentialMovingAverage(model, ema_decay) if ema_decay > 0.0 else None

    hindcast_start = float(os.getenv("V10_HINDCAST_WEIGHT_START", "0.50"))
    hindcast_end = float(os.getenv("V10_HINDCAST_WEIGHT_END", "0.15"))
    hindcast_decay_epochs = int(os.getenv("V10_HINDCAST_DECAY_EPOCHS", "20"))
    wind_aux_weight = float(os.getenv("V10_WIND_AUX_WEIGHT", "0.15"))
    transit_l2_weight = float(os.getenv("V10_TRANSIT_RESIDUAL_L2", "0.003"))
    surge_weight = float(os.getenv("V10_SURGE_WEIGHT", "0.02"))
    surge_pos_weight = torch.tensor(
        float(os.getenv("V10_SURGE_POS_WEIGHT", "2.3")), device=DEVICE
    )
    component_l2_weight = float(os.getenv("V10_COMPONENT_L2", "0.001"))
    patience = int(os.getenv("V10_EARLY_STOP_PATIENCE", "15"))

    manifest = {
        "train": chain_manifest(train_chains),
        "validation": chain_manifest(val_chains),
        "train_rows_used": int(len(selected_train_index)),
        "validation_rows_used": int(len(selected_val_index)),
        "chain_balanced_sampling": chain_balanced,
        "ar_validation_rmse_km_s": ar_val_micro,
        "ar_validation_chain_macro_rmse_km_s": ar_val_macro,
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
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"architecture={FILE_STEM} device={DEVICE} parameters={parameter_count:,} "
        f"wind_only={wind_only} image_size={IMAGE_SIZE} spatial_grid=2x4 "
        f"train_chains={train_chains.count} val_chains={val_chains.count} "
        f"chain_balanced={chain_balanced} lr={peak_lr:.2e} "
        f"physical_lr_mult={physical_multiplier:.1f} ema_decay={ema_decay} "
        f"mask={SOLAR_DISK_MASK} norm={IMAGE_NORM} loss=rmse_normalized"
    )

    best_val_rmse = float("inf")
    epochs_without_improvement = 0
    history = []
    val_target_rows = val_targets[selected_val_index]
    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        decay_progress = min(1.0, (epoch - 1) / max(1, hindcast_decay_epochs))
        hindcast_weight = hindcast_start + (
            hindcast_end - hindcast_start
        ) * decay_progress
        train_metrics = run_epoch(
            model,
            train_loader,
            train_chains.count,
            optimizer,
            scaler,
            True,
            hindcast_weight,
            wind_aux_weight,
            transit_l2_weight,
            surge_weight,
            surge_pos_weight,
            component_l2_weight,
            ema=ema,
        )
        validation_model = ema.model if ema is not None else model
        with torch.no_grad():
            val_metrics = run_epoch(
                validation_model,
                val_loader,
                val_chains.count,
                optimizer,
                scaler,
                False,
                hindcast_weight,
                wind_aux_weight,
                transit_l2_weight,
                surge_weight,
                surge_pos_weight,
                component_l2_weight,
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
            "train_wind_rmse_km_s": train_metrics["wind_rmse"],
            "val_wind_rmse_km_s": val_metrics["wind_rmse"],
            "val_ar_rmse_km_s": val_metrics["ar_rmse"],
            "train_hindcast_rmse_km_s": train_metrics["hindcast_rmse"],
            "val_hindcast_rmse_km_s": val_metrics["hindcast_rmse"],
            "val_surge_auroc": val_metrics["surge_auroc"],
            "val_wind_residual_rms_km_s": val_metrics["wind_residual_rms"],
            "val_propagation_rms_km_s": val_metrics["propagation_rms"],
            "val_correction_rms_km_s": val_metrics["correction_rms"],
            "hindcast_weight": hindcast_weight,
            "learning_rate": learning_rate,
            "seconds": elapsed,
            **{
                f"val_{name}": value
                for name, value in val_metrics["diagnostics"].items()
            },
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(
            f"epoch={epoch:03d} train_rmse={train_metrics['rmse']:.3f} "
            f"val_rmse={val_metrics['rmse']:.3f} "
            f"val_chain_macro_rmse={val_metrics['chain_macro_rmse']:.3f} "
            f"wind_val_rmse={val_metrics['wind_rmse']:.3f} "
            f"hind={val_metrics['hindcast_rmse']:.1f} "
            f"auroc={val_metrics['surge_auroc']:.3f} "
            f"lr={learning_rate:.2e} seconds={elapsed:.1f}",
            flush=True,
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "architecture": ARCHITECTURE_NAME,
                    "version": 10,
                    "model_state_dict": validation_model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "epoch": epoch,
                    "val_rmse_km_s": val_metrics["rmse"],
                    "val_chain_macro_rmse_km_s": val_metrics[
                        "chain_macro_rmse"
                    ],
                    "channels": CHANNELS,
                    "preprocess": {
                        "image_size": IMAGE_SIZE,
                        "image_norm": IMAGE_NORM,
                        "soft_cubic_strength": SOFT_CUBIC_STRENGTH,
                        "solar_disk_mask": SOLAR_DISK_MASK,
                        "solar_disk_radius_fraction": SOLAR_DISK_RADIUS_FRACTION,
                        "feature_schema": (
                            "seokho_v5b_masked_intensity_signed_delta4_"
                            "ar2_fixed96h_bounded_components_v1"
                        ),
                        "spatial_grid": [2, 4],
                        "ema_decay": ema_decay,
                        "warmup_epochs": warmup_epochs,
                        "minimum_learning_rate": minimum_lr,
                        "optimizer_weight_decay": weight_decay,
                    },
                },
                checkpoint_path,
            )
            save_validation_predictions(val_metrics, val_target_rows, validation_path)
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
    print(f"saved: {checkpoint_path.resolve()} best_val_rmse={best_val_rmse:.3f}")


if __name__ == "__main__":
    main()
