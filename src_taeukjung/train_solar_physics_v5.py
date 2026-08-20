import copy
import json
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

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
from model_solar_physics_v5 import SolarWindPhysicsTransformerV5


def fit_horizon_baseline(inputs, targets, chain_ids=None):
    last_wind = inputs[WIND_COLUMNS[-1]].to_numpy(np.float32) / 1000.0
    target = np.asarray(targets, dtype=np.float32) / 1000.0
    if chain_ids is None:
        weights = np.ones(len(last_wind), dtype=np.float32)
    else:
        chain_ids = np.asarray(chain_ids, dtype=np.int64)
        unique_ids, counts = np.unique(chain_ids, return_counts=True)
        count_by_id = dict(zip(unique_ids.tolist(), counts.tolist()))
        weights = np.asarray(
            [1.0 / count_by_id[int(chain_id)] for chain_id in chain_ids],
            dtype=np.float32,
        )
    weights = weights / weights.sum()

    wind_mean = np.sum(weights * last_wind)
    target_mean = np.sum(weights[:, None] * target, axis=0)
    centered_wind = last_wind - wind_mean
    centered_target = target - target_mean
    denominator = float(np.sum(weights * centered_wind**2))
    if denominator <= 0.0:
        raise ValueError("cannot fit baseline from constant wind inputs")
    slope = np.sum(
        weights[:, None] * centered_wind[:, None] * centered_target, axis=0
    ) / denominator
    intercept = target_mean - slope * wind_mean
    return slope.astype(np.float32), intercept.astype(np.float32)


def fit_baseline_residual_scale(inputs, targets, slope, intercept, chain_ids=None):
    last_wind = inputs[WIND_COLUMNS[-1]].to_numpy(np.float32) / 1000.0
    target = np.asarray(targets, dtype=np.float32) / 1000.0
    prediction = last_wind[:, None] * slope + intercept
    if chain_ids is None:
        weights = np.ones(len(last_wind), dtype=np.float32)
    else:
        chain_ids = np.asarray(chain_ids, dtype=np.int64)
        unique_ids, counts = np.unique(chain_ids, return_counts=True)
        count_by_id = dict(zip(unique_ids.tolist(), counts.tolist()))
        weights = np.asarray(
            [1.0 / count_by_id[int(chain_id)] for chain_id in chain_ids],
            dtype=np.float32,
        )
    weights = weights / weights.sum()
    residual = target - prediction
    residual_mean = np.sum(weights[:, None] * residual, axis=0)
    variance = np.sum(
        weights[:, None] * np.square(residual - residual_mean), axis=0
    )
    return np.sqrt(np.maximum(variance, 1e-8)).astype(np.float32)


class ExponentialMovingAverage:
    def __init__(self, model, decay):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be between 0 and 1")
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


def baseline_validation_metrics(inputs, targets, slope, intercept, chain_ids):
    last_wind = inputs[WIND_COLUMNS[-1]].to_numpy(np.float32) / 1000.0
    target = np.asarray(targets, dtype=np.float32) / 1000.0
    prediction = last_wind[:, None] * slope + intercept
    squared_error = ((prediction - target) * 1000.0) ** 2
    micro = float(np.sqrt(np.mean(squared_error)))
    chain_rmse = [
        float(np.sqrt(np.mean(squared_error[chain_ids == chain_id])))
        for chain_id in np.unique(chain_ids)
    ]
    return micro, float(np.mean(chain_rmse))


def run_epoch(
    model,
    loader,
    chain_count,
    optimizer,
    scaler,
    training,
    wind_aux_weight,
    residual_l2_weight=0.0,
    ema=None,
    collect_predictions=False,
):
    model.train(training)
    squared_error_sum = 0.0
    wind_squared_error_sum = 0.0
    wind_residual_squared_sum = 0.0
    fusion_residual_squared_sum = 0.0
    value_count = 0
    chain_squared_error_sum = np.zeros(chain_count, dtype=np.float64)
    chain_value_count = np.zeros(chain_count, dtype=np.int64)
    predictions = []
    sample_ids = []
    chain_ids_output = []
    chain_positions_output = []
    diagnostic_sums = {}
    diagnostic_sample_count = 0

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["images"].to(DEVICE, non_blocking=PIN_MEMORY)
        wind = batch["wind"].to(DEVICE, non_blocking=PIN_MEMORY)
        target = batch["target"].to(DEVICE, non_blocking=PIN_MEMORY)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(AMP_DEVICE_TYPE, enabled=USE_AMP):
                prediction, wind_prediction, wind_residual, fusion_residual = model(
                    images, wind, return_components=True
                )
            error_km_s = (prediction.float() - target.float()) * 1000.0
            wind_error_km_s = (wind_prediction.float() - target.float()) * 1000.0
            loss = torch.mean(error_km_s.square())
            loss = loss + wind_aux_weight * torch.mean(wind_error_km_s.square())
            if residual_l2_weight > 0.0:
                fusion_residual_km_s = fusion_residual.float() * 1000.0
                loss = loss + residual_l2_weight * torch.mean(
                    fusion_residual_km_s.square()
                )
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)

        detached_error = error_km_s.detach()
        detached_wind_error = wind_error_km_s.detach()
        wind_residual_km_s = wind_residual.detach().float() * 1000.0
        fusion_residual_km_s = fusion_residual.detach().float() * 1000.0
        squared_error_sum += float(detached_error.square().sum().cpu())
        wind_squared_error_sum += float(detached_wind_error.square().sum().cpu())
        wind_residual_squared_sum += float(wind_residual_km_s.square().sum().cpu())
        fusion_residual_squared_sum += float(fusion_residual_km_s.square().sum().cpu())
        value_count += detached_error.numel()

        if hasattr(model, "training_diagnostics"):
            batch_diagnostics = model.training_diagnostics()
            for name, value in batch_diagnostics.items():
                scalar = float(torch.as_tensor(value).detach().float().cpu())
                diagnostic_sums[name] = (
                    diagnostic_sums.get(name, 0.0) + scalar * images.size(0)
                )
            diagnostic_sample_count += images.size(0)

        batch_chain_ids = batch["chain_id"].numpy()
        row_squared_error = detached_error.square().sum(dim=1).cpu().numpy()
        np.add.at(chain_squared_error_sum, batch_chain_ids, row_squared_error)
        np.add.at(chain_value_count, batch_chain_ids, detached_error.shape[1])

        if collect_predictions:
            predictions.append(prediction.detach().float().cpu().numpy() * 1000.0)
            sample_ids.extend(batch["sample_id"])
            chain_ids_output.extend(batch_chain_ids.tolist())
            chain_positions_output.extend(batch["chain_position"].numpy().tolist())

        if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(loader):
            mode = "train" if training else "val"
            print(
                f"{mode} batch={batch_index}/{len(loader)} "
                f"running_rmse={math.sqrt(squared_error_sum / value_count):.3f} "
                f"wind_rmse={math.sqrt(wind_squared_error_sum / value_count):.3f} "
                f"image_residual_rms="
                f"{math.sqrt(fusion_residual_squared_sum / value_count):.3f}",
                flush=True,
            )

    valid_chains = chain_value_count > 0
    chain_rmse = np.sqrt(
        chain_squared_error_sum[valid_chains] / chain_value_count[valid_chains]
    )
    result = {
        "rmse": math.sqrt(squared_error_sum / value_count),
        "wind_rmse": math.sqrt(wind_squared_error_sum / value_count),
        "wind_residual_rms": math.sqrt(wind_residual_squared_sum / value_count),
        "fusion_residual_rms": math.sqrt(fusion_residual_squared_sum / value_count),
        "chain_macro_rmse": float(chain_rmse.mean()),
        "chain_rmse": chain_rmse,
    }
    if diagnostic_sample_count > 0:
        result["diagnostics"] = {
            name: total / diagnostic_sample_count
            for name, total in diagnostic_sums.items()
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
    frame = pd.DataFrame(
        metrics["predictions"],
        columns=[f"prediction_{column}" for column in TARGET_COLUMNS],
    )
    frame.insert(0, "chain_position", metrics["chain_positions"])
    frame.insert(0, "chain_id", metrics["chain_ids"])
    frame.insert(0, "sample_id", metrics["sample_ids"])
    actual = pd.DataFrame(
        targets,
        columns=[f"actual_{column}" for column in TARGET_COLUMNS],
    )
    pd.concat([frame, actual], axis=1).to_csv(path, index=False)


def main(
    model_class=SolarWindPhysicsTransformerV5,
    architecture_name="SolarWindPhysicsTransformerV5",
    version=5,
    file_stem="solar_physics_v5",
    feature_schema="cea_ch_quantiles_dark_area_ratio_delta_v1",
    extra_model_kwargs=None,
    training_image_flip_probability=0.0,
    residual_l2_weight=0.0,
    ema_decay=None,
    use_baseline_residual_scale=False,
):
    wind_only = os.getenv("WIND_ONLY", "0").lower() in {"1", "true", "yes"}
    chain_balanced = os.getenv("CHAIN_BALANCED_SAMPLING", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    latitude_bins = int(os.getenv("SOLAR_V5_LATITUDE_BINS", "4"))
    longitude_bins = int(os.getenv("SOLAR_V5_LONGITUDE_BINS", "8"))
    wind_aux_weight = float(os.getenv("SOLAR_V5_WIND_AUX_WEIGHT", "0.20"))
    architecture_kwargs = {
        key: SOLAR_PROBABILISTIC_KWARGS[key]
        for key in (
            "d_model",
            "wind_dim",
            "nhead",
            "encoder_layers",
            "ff_dim",
            "dropout",
        )
    }

    train_chains = infer_temporal_chains(train_inputs, IMAGE_COLUMNS)
    val_chains = infer_temporal_chains(val_inputs, IMAGE_COLUMNS)
    selected_train_index = limited_indexes(train_index, "MAX_TRAIN_SAMPLES")
    selected_val_index = limited_indexes(val_index, "MAX_VAL_SAMPLES")
    train_dataset = ChainAwareSolarWindDataset(
        train_image_array,
        train_image_index,
        train_inputs,
        selected_train_index,
        train_targets,
        temporal_chains=train_chains,
        north_south_flip_probability=training_image_flip_probability,
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

    manifest = {
        "train": chain_manifest(train_chains),
        "validation": chain_manifest(val_chains),
        "train_rows_used": int(len(selected_train_index)),
        "validation_rows_used": int(len(selected_val_index)),
        "chain_balanced_sampling": chain_balanced,
    }
    (OUTPUT_DIR / f"{file_stem}_chain_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    baseline_chain_ids = train_chains.chain_ids if chain_balanced else None
    baseline_slope, baseline_intercept = fit_horizon_baseline(
        train_inputs, train_targets, baseline_chain_ids
    )
    baseline_residual_scale = fit_baseline_residual_scale(
        train_inputs,
        train_targets,
        baseline_slope,
        baseline_intercept,
        baseline_chain_ids,
    )
    model_kwargs = {
        "image_size": IMAGE_SIZE,
        "apply_solar_disk_mask": SOLAR_DISK_MASK,
        "solar_disk_center_fraction": SOLAR_DISK_CENTER_FRACTION,
        "solar_disk_radius_fraction": SOLAR_DISK_RADIUS_FRACTION,
        "solar_cea_radius_fraction": SOLAR_CEA_RADIUS_FRACTION,
        "latitude_bins": latitude_bins,
        "longitude_bins": longitude_bins,
        "use_images": not wind_only,
        **architecture_kwargs,
        **(extra_model_kwargs or {}),
    }
    if use_baseline_residual_scale:
        model_kwargs["baseline_residual_scale"] = baseline_residual_scale.tolist()
    model = model_class(
        baseline_slope=baseline_slope,
        baseline_intercept=baseline_intercept,
        **model_kwargs,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=SOLAR_PROBABILISTIC_LR, weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.25, patience=3, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler(AMP_DEVICE_TYPE, enabled=USE_AMP)
    ema = ExponentialMovingAverage(model, ema_decay) if ema_decay is not None else None
    checkpoint_path = OUTPUT_DIR / f"best_{file_stem}.pth"
    history_path = OUTPUT_DIR / f"{file_stem}_history.csv"
    prediction_path = OUTPUT_DIR / f"{file_stem}_validation_predictions.csv"
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    val_rows = val_inputs.iloc[selected_val_index]
    val_target_rows = val_targets[selected_val_index]
    val_chain_ids = val_chains.chain_ids[selected_val_index]
    baseline_micro, baseline_macro = baseline_validation_metrics(
        val_rows,
        val_target_rows,
        baseline_slope,
        baseline_intercept,
        val_chain_ids,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"architecture={file_stem} device={DEVICE} parameters={parameter_count:,} "
        f"wind_only={wind_only} image_size={IMAGE_SIZE} "
        f"cea_grid={latitude_bins}x{longitude_bins} "
        f"train_chains={train_chains.count} val_chains={val_chains.count} "
        f"chain_balanced={chain_balanced} lr={SOLAR_PROBABILISTIC_LR:.2e} "
        f"wind_aux_weight={wind_aux_weight:.2f} "
        f"north_south_flip={training_image_flip_probability:.2f} "
        f"residual_l2={residual_l2_weight:.3f} "
        f"ema_decay={ema_decay} loss=mse_km_s"
    )
    print(
        f"linear_baseline_val_rmse={baseline_micro:.3f} "
        f"linear_baseline_chain_macro_rmse={baseline_macro:.3f}"
    )

    patience = 8
    best_val_rmse = float("inf")
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            train_chains.count,
            optimizer,
            scaler,
            training=True,
            wind_aux_weight=wind_aux_weight,
            residual_l2_weight=residual_l2_weight,
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
                training=False,
                wind_aux_weight=wind_aux_weight,
                residual_l2_weight=residual_l2_weight,
                collect_predictions=True,
            )

        scheduler.step(val_metrics["rmse"])
        learning_rate = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - started
        history.append(
            {
                "epoch": epoch,
                "train_rmse_km_s": train_metrics["rmse"],
                "val_rmse_km_s": val_metrics["rmse"],
                "train_chain_macro_rmse_km_s": train_metrics["chain_macro_rmse"],
                "val_chain_macro_rmse_km_s": val_metrics["chain_macro_rmse"],
                "train_wind_rmse_km_s": train_metrics["wind_rmse"],
                "val_wind_rmse_km_s": val_metrics["wind_rmse"],
                "train_image_residual_rms_km_s": train_metrics[
                    "fusion_residual_rms"
                ],
                "val_image_residual_rms_km_s": val_metrics[
                    "fusion_residual_rms"
                ],
                "learning_rate": learning_rate,
                "seconds": elapsed,
                **{
                    f"train_{name}": value
                    for name, value in train_metrics.get("diagnostics", {}).items()
                },
                **{
                    f"val_{name}": value
                    for name, value in val_metrics.get("diagnostics", {}).items()
                },
            }
        )
        pd.DataFrame(history).to_csv(history_path, index=False)
        diagnostic_text = "".join(
            f" {name}={value:.3f}"
            for name, value in val_metrics.get("diagnostics", {}).items()
        )
        print(
            f"epoch={epoch:03d} train_rmse={train_metrics['rmse']:.3f} "
            f"val_rmse={val_metrics['rmse']:.3f} "
            f"val_chain_macro_rmse={val_metrics['chain_macro_rmse']:.3f} "
            f"wind_val_rmse={val_metrics['wind_rmse']:.3f} "
            f"lr={learning_rate:.2e} seconds={elapsed:.1f}"
            f"{diagnostic_text}",
            flush=True,
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "architecture": architecture_name,
                    "version": version,
                    "model_state_dict": validation_model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "epoch": epoch,
                    "val_rmse_km_s": val_metrics["rmse"],
                    "val_chain_macro_rmse_km_s": val_metrics["chain_macro_rmse"],
                    "channels": CHANNELS,
                    "preprocess": {
                        "image_size": IMAGE_SIZE,
                        "image_norm": IMAGE_NORM,
                        "soft_cubic_strength": SOFT_CUBIC_STRENGTH,
                        "solar_disk_mask": SOLAR_DISK_MASK,
                        "solar_disk_radius_fraction": SOLAR_DISK_RADIUS_FRACTION,
                        "solar_cea_radius_fraction": SOLAR_CEA_RADIUS_FRACTION,
                        "feature_schema": feature_schema,
                        "cea_grid": [latitude_bins, longitude_bins],
                        "north_south_flip_probability": training_image_flip_probability,
                        "residual_l2_weight": residual_l2_weight,
                        "ema_decay": ema_decay,
                    },
                },
                checkpoint_path,
            )
            save_validation_predictions(
                val_metrics, val_target_rows, prediction_path
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
    plt.savefig(OUTPUT_DIR / f"{file_stem}_learning_curve.png", dpi=140)
    print(f"saved: {checkpoint_path.resolve()} best_val_rmse={best_val_rmse:.3f}")


if __name__ == "__main__":
    main()
