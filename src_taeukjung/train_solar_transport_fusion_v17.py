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
from model_solar_transport_fusion_v17 import (
    ARCHITECTURE_NAME,
    FILE_STEM,
    SolarWindTransportFusionV17,
)
from train_solar_hybrid_v10 import limited_indexes, save_validation_predictions


FEATURE_SCHEMA = "native_latband_longitude_fixed_speed_transport_ar_fusion_v17"
CHECKPOINT_VERSION = "17"
MODEL_CHANGES = [
    "native 64-column north-center-south profiles without image resizing",
    "fixed 300/400/500/650/800 km/s experts lock source value to delay",
    "causal observed-wind transport pretraining before target supervision",
    "frozen-transport AR(2) fusion followed by low-LR joint fine-tuning",
    "direct transport-minus-AR correction without a learnable evidence gate",
    "matched scrambled-image and no-pretraining controls",
]


def boolean_environment(name, default=False):
    fallback = "1" if default else "0"
    return os.getenv(name, fallback).lower() in {"1", "true", "yes"}


def parse_float_tuple(name, default):
    value = os.getenv(name, default)
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def fit_ar_configuration(train_chains, val_chains):
    order = int(os.getenv("V17_AR_ORDER", "2"))
    ridge_strength = float(os.getenv("V17_AR_RIDGE", "30"))
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
        "baseline_residual_scale": ar_residual_scale.tolist(),
        "column_dim": int(os.getenv("V17_COLUMN_DIM", "32")),
        "longitude_kernel_size": int(
            os.getenv("V17_LONGITUDE_KERNEL_SIZE", "5")
        ),
        "speed_experts_kms": parse_float_tuple(
            "V17_SPEED_EXPERTS_KMS", "300,400,500,650,800"
        ),
        "transport_sigma_hours": float(
            os.getenv("V17_TRANSPORT_SIGMA_HOURS", "15")
        ),
        "effective_distance_hours_at_1000_kms": float(
            os.getenv("V17_EFFECTIVE_DISTANCE_HOURS_AT_1000_KMS", "41.6")
        ),
        "solar_rotation_days": float(os.getenv("V17_SOLAR_ROTATION_DAYS", "27.27")),
        "minimum_delay_hours": float(
            os.getenv("V17_MINIMUM_DELAY_HOURS", "48")
        ),
        "maximum_delay_hours": float(
            os.getenv("V17_MAXIMUM_DELAY_HOURS", "144")
        ),
        "transport_strength": float(os.getenv("V17_TRANSPORT_STRENGTH", "0.50")),
        "correction_cap_multiplier": float(
            os.getenv("V17_CORRECTION_CAP_MULTIPLIER", "1.0")
        ),
        "dropout": float(os.getenv("V17_DROPOUT", "0.10")),
        "time_mask_prob": float(os.getenv("V17_TIME_MASK_PROBABILITY", "0.10")),
        "modality_drop_prob": float(
            os.getenv("V17_MODALITY_DROP_PROBABILITY", "0.10")
        ),
        "delta_gain": float(os.getenv("V17_DELTA_GAIN", "1.0")),
        "scramble_images": boolean_environment("V17_SCRAMBLE_IMAGES"),
        "apply_solar_disk_mask": SOLAR_DISK_MASK,
        "solar_disk_center_fraction": SOLAR_DISK_CENTER_FRACTION,
        "solar_disk_radius_fraction": SOLAR_DISK_RADIUS_FRACTION,
        "solar_disk_edge_pixels": float(
            os.getenv("V17_SOLAR_DISK_EDGE_PIXELS", "1.5")
        ),
    }


def v17_preprocess(model_kwargs=None):
    kwargs = model_kwargs or {}
    return {
        "image_size": IMAGE_SIZE,
        "image_norm": IMAGE_NORM,
        "soft_cubic_strength": SOFT_CUBIC_STRENGTH,
        "solar_disk_mask": SOLAR_DISK_MASK,
        "solar_disk_radius_fraction": SOLAR_DISK_RADIUS_FRACTION,
        "feature_schema": FEATURE_SCHEMA,
        "native_longitude_columns": IMAGE_SIZE,
        "latitude_bands": ["north", "center", "south"],
        "profile_statistics": ["mean", "std", "dark_fraction", "bright_fraction"],
        "speed_experts_kms": list(
            kwargs.get(
                "speed_experts_kms",
                parse_float_tuple("V17_SPEED_EXPERTS_KMS", "300,400,500,650,800"),
            )
        ),
        "transport_sigma_hours": float(
            kwargs.get(
                "transport_sigma_hours",
                os.getenv("V17_TRANSPORT_SIGMA_HOURS", "15"),
            )
        ),
        "scramble_images": bool(
            kwargs.get(
                "scramble_images", boolean_environment("V17_SCRAMBLE_IMAGES")
            )
        ),
    }


def masked_rmse(prediction, target, sample_mask):
    mask = sample_mask.to(dtype=prediction.dtype).view(-1, 1)
    denominator = (mask.sum() * prediction.shape[1]).clamp_min(1.0)
    return torch.sqrt(
        ((prediction - target).square() * mask).sum() / denominator
        + RMSE_EPSILON
    )


def run_epoch(
    model,
    loader,
    chain_count,
    optimizer,
    scaler,
    training,
    stage,
    hindcast_weight,
    future_transport_weight,
    correction_l2_weight,
    smoothness_weight,
    entropy_weight,
    gradient_clip,
    collect_predictions=False,
):
    model.train(training)
    totals = {
        "forecast": 0.0,
        "ar_base": 0.0,
        "correction": 0.0,
        "transport_hindcast": 0.0,
        "transport_future": 0.0,
    }
    forecast_count = 0
    hindcast_count = 0
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
                    images, wind, return_components=True, return_aux=True
                )
                forecast_loss = torch.sqrt(
                    F.mse_loss(prediction, target) + RMSE_EPSILON
                )
                transport_hindcast_loss = masked_rmse(
                    aux["transport_hindcast"],
                    aux["hindcast_wind"],
                    aux["image_keep"],
                )
                transport_future_loss = masked_rmse(
                    aux["transport_forecast"], target, aux["image_keep"]
                )
                correction_l2 = components["image_correction"].square().mean()
                expert_probability = aux["expert_probability"]
                temporal_smoothness = (
                    expert_probability[:, 1:] - expert_probability[:, :-1]
                ).square().mean()
                longitude_smoothness = (
                    expert_probability[:, :, 1:] - expert_probability[:, :, :-1]
                ).square().mean()
                smoothness = temporal_smoothness + longitude_smoothness
                expert_entropy = -(
                    expert_probability.clamp_min(1e-8).log()
                    * expert_probability
                ).sum(dim=-1).mean()

                if stage == "transport":
                    loss = (
                        transport_hindcast_loss
                        + smoothness_weight * smoothness
                        + entropy_weight * expert_entropy
                    )
                elif stage == "fusion":
                    loss = forecast_loss + correction_l2_weight * correction_l2
                elif stage == "joint":
                    loss = (
                        forecast_loss
                        + hindcast_weight * transport_hindcast_loss
                        + future_transport_weight * transport_future_loss
                        + correction_l2_weight * correction_l2
                        + smoothness_weight * smoothness
                        + entropy_weight * expert_entropy
                    )
                else:
                    raise ValueError(f"unknown stage: {stage}")

            if training:
                scaler.scale(loss).backward()
                if gradient_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        ],
                        gradient_clip,
                    )
                scaler.step(optimizer)
                scaler.update()

        forecast_error = (prediction.detach().float() - target.float()) * 1000.0
        ar_error = (components["ar_base"].detach().float() - target.float()) * 1000.0
        correction = components["image_correction"].detach().float() * 1000.0
        hindcast_error = (
            aux["transport_hindcast"].detach().float()
            - aux["hindcast_wind"].float()
        ) * 1000.0
        transport_future_error = (
            aux["transport_forecast"].detach().float() - target.float()
        ) * 1000.0
        totals["forecast"] += float(forecast_error.square().sum().cpu())
        totals["ar_base"] += float(ar_error.square().sum().cpu())
        totals["correction"] += float(correction.square().sum().cpu())
        totals["transport_hindcast"] += float(
            hindcast_error.square().sum().cpu()
        )
        totals["transport_future"] += float(
            transport_future_error.square().sum().cpu()
        )
        forecast_count += forecast_error.numel()
        hindcast_count += hindcast_error.numel()
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
                f"{mode} stage={stage} batch={batch_index}/{len(loader)} "
                f"rmse={math.sqrt(totals['forecast'] / forecast_count):.3f} "
                f"ar_rmse={math.sqrt(totals['ar_base'] / forecast_count):.3f} "
                f"transport_hind={math.sqrt(totals['transport_hindcast'] / hindcast_count):.3f} "
                f"transport_future={math.sqrt(totals['transport_future'] / forecast_count):.3f} "
                f"corr_rms={math.sqrt(totals['correction'] / forecast_count):.3f}",
                flush=True,
            )

    valid_chains = chain_value_count > 0
    result = {
        "rmse": math.sqrt(totals["forecast"] / forecast_count),
        "chain_macro_rmse": float(
            np.sqrt(
                chain_squared_error[valid_chains]
                / chain_value_count[valid_chains]
            ).mean()
        ),
        "ar_base_rmse": math.sqrt(totals["ar_base"] / forecast_count),
        "correction_rms": math.sqrt(totals["correction"] / forecast_count),
        "transport_hindcast_rmse": math.sqrt(
            totals["transport_hindcast"] / hindcast_count
        ),
        "transport_future_rmse": math.sqrt(
            totals["transport_future"] / forecast_count
        ),
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


def make_optimizer(model, stage):
    learning_rates = {
        "transport": float(os.getenv("V17_TRANSPORT_LR", "3e-4")),
        "fusion": float(os.getenv("V17_FUSION_LR", "3e-4")),
        "joint": float(os.getenv("V17_JOINT_LR", "5e-5")),
    }
    weight_decay = float(os.getenv("V17_WEIGHT_DECAY", "0.02"))
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError(f"stage {stage} has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rates[stage], weight_decay=weight_decay
    )
    return optimizer, learning_rates[stage], weight_decay


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
    model = SolarWindTransportFusionV17(**model_kwargs).to(DEVICE)
    phases = [
        ("transport", int(os.getenv("V17_TRANSPORT_EPOCHS", "6"))),
        ("fusion", int(os.getenv("V17_FUSION_EPOCHS", "10"))),
        ("joint", int(os.getenv("V17_JOINT_EPOCHS", "4"))),
    ]
    if wind_only:
        raise ValueError("V17 is an image-transport experiment; use the reported AR baseline")
    if sum(epoch_count for _, epoch_count in phases) <= 0:
        raise ValueError("at least one V17 training epoch is required")

    hindcast_weight = float(os.getenv("V17_HINDCAST_WEIGHT", "0.50"))
    future_transport_weight = float(
        os.getenv("V17_FUTURE_TRANSPORT_WEIGHT", "0.10")
    )
    correction_l2_weight = float(
        os.getenv("V17_CORRECTION_L2_WEIGHT", "0.01")
    )
    smoothness_weight = float(os.getenv("V17_SMOOTHNESS_WEIGHT", "0.01"))
    entropy_weight = float(os.getenv("V17_ENTROPY_WEIGHT", "0.001"))
    gradient_clip = float(os.getenv("V17_GRADIENT_CLIP", "1.0"))
    patience = int(os.getenv("V17_EARLY_STOP_PATIENCE", "5"))
    minimum_lr = float(os.getenv("V17_MIN_LR", "1e-6"))

    manifest = {
        "train": chain_manifest(train_chains),
        "validation": chain_manifest(val_chains),
        "train_rows_used": int(len(selected_train_index)),
        "validation_rows_used": int(len(selected_val_index)),
        "training_phases": dict(phases),
        "source_models": [
            "train-only AR(2)",
            "Seokho speed-to-arrival and source-value findings",
        ],
        "v17_changes": MODEL_CHANGES,
    }
    (OUTPUT_DIR / f"{FILE_STEM}_chain_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    checkpoint_path = OUTPUT_DIR / f"best_{FILE_STEM}.pth"
    history_path = OUTPUT_DIR / f"{FILE_STEM}_history.csv"
    validation_path = OUTPUT_DIR / f"{FILE_STEM}_validation_predictions.csv"
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"architecture={FILE_STEM} device={DEVICE} parameters={parameter_count:,} "
        f"image_size={IMAGE_SIZE} native_columns={model.image_size} "
        f"speed_experts_kms={(model.speed_experts * 1000.0).tolist()} "
        f"delay_bounds=[{model.minimum_delay_hours:.0f},{model.maximum_delay_hours:.0f}] "
        f"sigma_h={model.transport_sigma_hours:.1f} phases={dict(phases)} "
        f"train_chains={train_chains.count} val_chains={val_chains.count} "
        f"scramble={model.scramble_images} mask={SOLAR_DISK_MASK} norm={IMAGE_NORM} "
        f"loss=staged_transport_hindcast_then_ar_fusion_then_joint"
    )

    best_val_rmse = float("inf")
    history = []
    global_epoch = 0
    val_target_rows = val_targets[selected_val_index]
    for stage, stage_epochs in phases:
        if stage_epochs <= 0:
            continue
        model.set_stage(stage)
        optimizer, peak_lr, weight_decay = make_optimizer(model, stage)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, stage_epochs), eta_min=minimum_lr
        )
        scaler = torch.amp.GradScaler(AMP_DEVICE_TYPE, enabled=USE_AMP)
        stage_best = float("inf")
        stage_without_improvement = 0
        trainable_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        print(
            f"\n[stage] {stage}: epochs={stage_epochs} lr={peak_lr:.2e} "
            f"trainable={trainable_count:,}"
        )
        for stage_epoch in range(1, stage_epochs + 1):
            global_epoch += 1
            started = time.perf_counter()
            train_metrics = run_epoch(
                model,
                train_loader,
                train_chains.count,
                optimizer,
                scaler,
                True,
                stage,
                hindcast_weight,
                future_transport_weight,
                correction_l2_weight,
                smoothness_weight,
                entropy_weight,
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
                    stage,
                    hindcast_weight,
                    future_transport_weight,
                    correction_l2_weight,
                    smoothness_weight,
                    entropy_weight,
                    gradient_clip,
                    collect_predictions=True,
                )
            learning_rate = optimizer.param_groups[0]["lr"]
            scheduler.step()
            elapsed = time.perf_counter() - started
            row = {
                "epoch": global_epoch,
                "stage": stage,
                "stage_epoch": stage_epoch,
                "train_rmse_km_s": train_metrics["rmse"],
                "val_rmse_km_s": val_metrics["rmse"],
                "val_chain_macro_rmse_km_s": val_metrics["chain_macro_rmse"],
                "val_ar_base_rmse_km_s": val_metrics["ar_base_rmse"],
                "train_transport_hindcast_rmse_km_s": train_metrics[
                    "transport_hindcast_rmse"
                ],
                "val_transport_hindcast_rmse_km_s": val_metrics[
                    "transport_hindcast_rmse"
                ],
                "val_transport_future_rmse_km_s": val_metrics[
                    "transport_future_rmse"
                ],
                "val_image_correction_rms_km_s": val_metrics["correction_rms"],
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
                f"epoch={global_epoch:03d} stage={stage} stage_epoch={stage_epoch:02d} "
                f"train_rmse={train_metrics['rmse']:.3f} "
                f"val_rmse={val_metrics['rmse']:.3f} "
                f"val_chain_macro={val_metrics['chain_macro_rmse']:.3f} "
                f"transport_hind={val_metrics['transport_hindcast_rmse']:.3f} "
                f"transport_future={val_metrics['transport_future_rmse']:.3f} "
                f"corr_rms={val_metrics['correction_rms']:.3f} "
                f"delay_h={diagnostics['transport_expected_delay_h']:.1f} "
                f"expert_entropy={diagnostics['expert_entropy']:.3f} "
                f"lr={learning_rate:.2e} seconds={elapsed:.1f}",
                flush=True,
            )

            stage_metric = (
                val_metrics["transport_hindcast_rmse"]
                if stage == "transport"
                else val_metrics["rmse"]
            )
            if stage_metric < stage_best:
                stage_best = stage_metric
                stage_without_improvement = 0
            else:
                stage_without_improvement += 1

            if stage != "transport" and val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                torch.save(
                    {
                        "architecture": ARCHITECTURE_NAME,
                        "version": CHECKPOINT_VERSION,
                        "model_state_dict": model.state_dict(),
                        "model_kwargs": model_kwargs,
                        "epoch": global_epoch,
                        "stage": stage,
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
                            **v17_preprocess(model_kwargs),
                            "training_phases": dict(phases),
                            "optimizer_weight_decay": weight_decay,
                        },
                    },
                    checkpoint_path,
                )
                save_validation_predictions(
                    val_metrics, val_target_rows, validation_path
                )

            if patience > 0 and stage_without_improvement >= patience:
                print(f"early stopping stage={stage}")
                break

    if not checkpoint_path.exists():
        raise RuntimeError("V17 completed without a forecast-stage checkpoint")
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
