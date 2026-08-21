import json
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

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
from model_solar_source_map_v11_2 import (
    ARCHITECTURE_NAME,
    FILE_STEM,
    SolarWindSourceMapV11_2,
)
from train_solar_hybrid_v10 import limited_indexes, rank_auc, save_validation_predictions


FEATURE_SCHEMA = "seokho_v7_source_map_dynamic_grid_maskfix_v11_2"


class ConsecutivePairDataset(Dataset):
    """Attach the next selected row in each recovered temporal chain."""

    def __init__(self, dataset, temporal_chains, validate_overlap=True):
        self.dataset = dataset
        self.indexes = dataset.indexes
        self.chain_ids = dataset.chain_ids
        self.chain_positions = dataset.chain_positions
        row_to_item = {
            int(row_index): item
            for item, row_index in enumerate(self.indexes.tolist())
        }
        successor_by_row = {}
        for chain in temporal_chains.chains:
            for row_index, successor in zip(chain[:-1], chain[1:]):
                successor_by_row[int(row_index)] = int(successor)
        self.successor_items = np.full(len(self.indexes), -1, dtype=np.int64)
        for item, row_index in enumerate(self.indexes.tolist()):
            successor = successor_by_row.get(int(row_index))
            if successor in row_to_item:
                self.successor_items[item] = row_to_item[successor]
        if validate_overlap:
            self._validate_overlap()

    def _validate_overlap(self):
        base = self.dataset.base_dataset
        pair_items = np.flatnonzero(self.successor_items >= 0)
        for item in pair_items:
            successor_item = int(self.successor_items[item])
            row_index = int(self.indexes[item])
            successor_row = int(self.indexes[successor_item])
            if not np.array_equal(
                base.image_indexes[row_index, 1:],
                base.image_indexes[successor_row, :-1],
            ):
                raise ValueError("consecutive image windows do not overlap")
            if not np.allclose(
                base.wind[row_index, 1:],
                base.wind[successor_row, :-1],
                atol=1e-6,
                rtol=0.0,
            ):
                raise ValueError("consecutive wind windows do not overlap")
            if base.targets is not None and not np.allclose(
                base.targets[row_index, 1:],
                base.targets[successor_row, :-1],
                atol=1e-4,
                rtol=0.0,
            ):
                raise ValueError("consecutive target windows do not overlap")
        self.pair_count = int(len(pair_items))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        anchor = self.dataset[item]
        successor_item = int(self.successor_items[item])
        if successor_item >= 0:
            successor = self.dataset[successor_item]
            anchor["has_successor"] = True
            anchor["successor_images"] = successor["images"]
            anchor["successor_wind"] = successor["wind"]
        else:
            anchor["has_successor"] = False
            anchor["successor_images"] = anchor["images"]
            anchor["successor_wind"] = anchor["wind"]
        return anchor


def build_model_kwargs(wind_only=False):
    return {
        "image_size": IMAGE_SIZE,
        "use_images": not wind_only,
        "d_model": int(os.getenv("V112_D_MODEL", "128")),
        "dropout": float(os.getenv("V112_DROPOUT", "0.10")),
        "time_mask_prob": float(
            os.getenv("V112_TIME_MASK_PROBABILITY", "0.15")
        ),
        "modality_drop_prob": float(
            os.getenv("V112_MODALITY_DROP_PROBABILITY", "0.25")
        ),
        "delta_gain": float(os.getenv("V112_DELTA_GAIN", "1.0")),
        "grid_rows": int(os.getenv("V112_GRID_ROWS", "2")),
        "grid_columns": int(os.getenv("V112_GRID_COLUMNS", "4")),
        "apply_solar_disk_mask": SOLAR_DISK_MASK,
        "solar_disk_center_fraction": SOLAR_DISK_CENTER_FRACTION,
        "solar_disk_radius_fraction": SOLAR_DISK_RADIUS_FRACTION,
        "solar_disk_edge_pixels": float(
            os.getenv("V112_SOLAR_DISK_EDGE_PIXELS", "1.5")
        ),
        "kernel_sigma_hours": float(
            os.getenv("V112_KERNEL_SIGMA_HOURS", "12")
        ),
        "transit_residual_hours": float(
            os.getenv("V112_TRANSIT_RESIDUAL_HOURS", "24")
        ),
    }


def is_physical_parameter(name):
    return name in {
        "dist_eff_raw",
        "reversion_logit",
        "climatology",
        "fallback_weight_raw",
        "source_speed_head.bias",
        "source_gate_head.bias",
        "transit_residual_head.bias",
        "lon_offset_head.bias",
    }


def make_optimizer(model):
    peak_lr = float(os.getenv("LEARNING_RATE", "3e-5"))
    weight_decay = float(os.getenv("V112_WEIGHT_DECAY", "0.01"))
    physical_multiplier = float(os.getenv("V112_PHYSICAL_LR_MULT", "100"))
    base_parameters = []
    physical_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = (
            physical_parameters if is_physical_parameter(name) else base_parameters
        )
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
    warmup_epochs = int(os.getenv("V112_WARMUP_EPOCHS", "3"))
    minimum_lr = float(os.getenv("V112_MIN_LR", "1e-6"))
    if not 0 < warmup_epochs < EPOCHS:
        raise ValueError("V112_WARMUP_EPOCHS must be between 1 and EPOCHS - 1")

    def learning_rate_factor(step):
        if step < warmup_epochs:
            return (step + 1) / warmup_epochs
        progress = (step - warmup_epochs) / max(1, EPOCHS - warmup_epochs)
        floor = minimum_lr / peak_lr
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, learning_rate_factor
    )
    return scheduler, warmup_epochs, minimum_lr


def alignment_kl(model, aux, wind, target, sigma_deg):
    with torch.autocast(AMP_DEVICE_TYPE, enabled=False):
        source_weight = aux["source_weight"].float()
        batch_size = source_weight.shape[0]
        distance = model.effective_distance().detach().float()
        time_grid = torch.cat([model.hindcast_hours, model.horizon_hours]).float()
        y_series = torch.cat([wind[:, 7:], target], dim=1).float()
        transit_hours = distance / y_series.clamp(0.2, 1.2)
        delta_time = (
            time_grid.view(1, 1, -1)
            - transit_hours.unsqueeze(1)
            + model.image_age_hours.float().view(1, -1, 1)
        )
        longitude = -model.omega_deg_per_hour * delta_time
        visible = longitude.abs() <= 90.0
        gaussian = torch.exp(
            -(
                model.cell_lon_deg.float().view(
                    1, 1, model.grid_columns, 1
                )
                - longitude.unsqueeze(2)
            ).square()
            / (2.0 * float(sigma_deg) ** 2)
        ) * visible.unsqueeze(2)
        target_distribution = gaussian.unsqueeze(2).expand(
            -1, -1, model.grid_rows, -1, -1
        ) / float(model.grid_rows)
        target_distribution = target_distribution * aux["time_keep"].float().view(
            batch_size, -1, 1, 1, 1
        )
        target_mass = target_distribution.sum(dim=(1, 2, 3))
        valid = (
            (target_mass > 1e-6).float()
            * aux["image_keep"].float().unsqueeze(-1)
        )
        target_distribution = target_distribution / target_mass.clamp_min(
            1e-6
        ).view(batch_size, 1, 1, 1, -1)
        learned_distribution = source_weight + 1e-8
        learned_distribution = learned_distribution / learned_distribution.sum(
            dim=(1, 2, 3), keepdim=True
        )
        kl = (
            target_distribution
            * (
                target_distribution.clamp_min(1e-12).log()
                - learned_distribution.log()
            )
        ).sum(dim=(1, 2, 3))
        return (kl * valid).sum() / valid.sum().clamp_min(1.0)


def _first_half(values, batch_size):
    return {
        name: value[:batch_size] if value is not None else None
        for name, value in values.items()
    }


def run_epoch(
    model,
    loader,
    chain_count,
    optimizer,
    scaler,
    training,
    hindcast_weight,
    transit_l2_weight,
    alignment_weight,
    alignment_sigma_deg,
    surge_weight,
    surge_pos_weight,
    consistency_weight,
    collect_predictions=False,
):
    model.train(training)
    totals = {
        "forecast": 0.0,
        "base": 0.0,
        "propagation": 0.0,
        "hindcast": 0.0,
        "alignment": 0.0,
        "consistency": 0.0,
    }
    counts = {
        "forecast": 0,
        "hindcast": 0,
        "alignment": 0,
        "consistency": 0,
        "pairs": 0,
    }
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
        has_pairs = consistency_weight > 0.0 and "has_successor" in batch
        if has_pairs:
            successor_images = batch["successor_images"].to(
                DEVICE, non_blocking=PIN_MEMORY
            )
            successor_wind = batch["successor_wind"].to(
                DEVICE, non_blocking=PIN_MEMORY
            )
            pair_mask = batch["has_successor"].to(
                DEVICE, non_blocking=PIN_MEMORY
            ).bool()
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(AMP_DEVICE_TYPE, enabled=USE_AMP):
                if has_pairs:
                    batch_size = images.shape[0]
                    joined_images = torch.cat([images, successor_images], dim=0)
                    joined_wind = torch.cat([wind, successor_wind], dim=0)
                    time_keep, image_keep = model.sample_paired_augmentation_masks(
                        batch_size,
                        joined_images.device,
                        joined_images.dtype,
                    )
                    joined_prediction, joined_components, joined_aux = model(
                        joined_images,
                        joined_wind,
                        return_components=True,
                        return_aux=True,
                        time_keep=time_keep,
                        image_keep=image_keep,
                    )
                    prediction = joined_prediction[:batch_size]
                    successor_prediction = joined_prediction[batch_size:]
                    components = _first_half(joined_components, batch_size)
                    aux = _first_half(joined_aux, batch_size)
                else:
                    prediction, components, aux = model(
                        images, wind, return_components=True, return_aux=True
                    )

                loss = torch.sqrt(
                    F.mse_loss(prediction, target) + RMSE_EPSILON
                )
                if has_pairs and pair_mask.any():
                    consistency_error = (
                        prediction[pair_mask, 1:]
                        - successor_prediction[pair_mask, :-1]
                    )
                    consistency_loss = torch.sqrt(
                        consistency_error.square().mean() + RMSE_EPSILON
                    )
                    loss = loss + consistency_weight * consistency_loss
                else:
                    consistency_error = None

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
                    align_value = alignment_kl(
                        model, aux, wind, target, alignment_sigma_deg
                    )
                    loss = loss + alignment_weight * align_value
                else:
                    align_value = prediction.new_zeros(())

                surge_label = (
                    (target.max(dim=1).values - wind[:, -1])
                    > float(os.getenv("V112_SURGE_THRESHOLD_KMS", "100"))
                    / 1000.0
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

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        forecast_error = (prediction.detach().float() - target.float()) * 1000.0
        base_error = (components["base"].detach().float() - target.float()) * 1000.0
        propagation = (
            components["propagation_residual"].detach().float() * 1000.0
        )
        totals["forecast"] += float(forecast_error.square().sum().cpu())
        totals["base"] += float(base_error.square().sum().cpu())
        totals["propagation"] += float(propagation.square().sum().cpu())
        totals["alignment"] += float(align_value.detach().cpu())
        counts["forecast"] += forecast_error.numel()
        counts["alignment"] += 1

        if consistency_error is not None:
            consistency_error_kms = consistency_error.detach().float() * 1000.0
            totals["consistency"] += float(
                consistency_error_kms.square().sum().cpu()
            )
            counts["consistency"] += consistency_error_kms.numel()
            counts["pairs"] += int(pair_mask.sum().cpu())

        if aux["hindcast"] is not None:
            keep = aux["image_keep"].unsqueeze(-1)
            hindcast_error = (aux["hindcast"].detach() - wind[:, 7:]) * 1000.0
            totals["hindcast"] += float(
                (hindcast_error.square() * keep).sum().cpu()
            )
            counts["hindcast"] += int(
                (keep.sum() * hindcast_error.shape[1]).cpu()
            )

        batch_chain_ids = batch["chain_id"].numpy()
        row_squared_error = forecast_error.square().sum(dim=1).cpu().numpy()
        np.add.at(chain_squared_error, batch_chain_ids, row_squared_error)
        np.add.at(chain_value_count, batch_chain_ids, forecast_error.shape[1])

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
            consistency_text = (
                f" consistency_rmse="
                f"{math.sqrt(totals['consistency'] / counts['consistency']):.3f}"
                if counts["consistency"] > 0
                else ""
            )
            print(
                f"{mode} batch={batch_index}/{len(loader)} "
                f"running_rmse={math.sqrt(totals['forecast'] / counts['forecast']):.3f} "
                f"base_rmse={math.sqrt(totals['base'] / counts['forecast']):.3f} "
                f"prop_rms={math.sqrt(totals['propagation'] / counts['forecast']):.3f} "
                f"align_kl={totals['alignment'] / counts['alignment']:.3f}"
                f"{consistency_text}",
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
        "base_rmse": math.sqrt(totals["base"] / counts["forecast"]),
        "propagation_rms": math.sqrt(
            totals["propagation"] / counts["forecast"]
        ),
        "hindcast_rmse": (
            math.sqrt(totals["hindcast"] / counts["hindcast"])
            if counts["hindcast"] > 0
            else float("nan")
        ),
        "alignment_kl": totals["alignment"] / counts["alignment"],
        "consistency_rmse": (
            math.sqrt(totals["consistency"] / counts["consistency"])
            if counts["consistency"] > 0
            else float("nan")
        ),
        "consistency_pairs": counts["pairs"],
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


def _boolean_environment(name, default=False):
    fallback = "1" if default else "0"
    return os.getenv(name, fallback).lower() in {"1", "true", "yes"}


def main():
    wind_only = _boolean_environment("WIND_ONLY")
    consistency_weight = float(os.getenv("V112_CONSISTENCY_WEIGHT", "0"))
    if consistency_weight < 0.0:
        raise ValueError("V112_CONSISTENCY_WEIGHT must be nonnegative")
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
    pair_counts = {"train": 0, "validation": 0}
    if consistency_weight > 0.0:
        train_dataset = ConsecutivePairDataset(train_dataset, train_chains)
        val_dataset = ConsecutivePairDataset(val_dataset, val_chains)
        pair_counts = {
            "train": train_dataset.pair_count,
            "validation": val_dataset.pair_count,
        }
    chain_balanced = _boolean_environment("V112_CHAIN_BALANCED")
    train_loader = make_chain_loader(
        train_dataset, training=True, chain_balanced=chain_balanced
    )
    val_loader = make_chain_loader(val_dataset, training=False)

    model_kwargs = build_model_kwargs(wind_only=wind_only)
    model = SolarWindSourceMapV11_2(**model_kwargs).to(DEVICE)
    optimizer, peak_lr, weight_decay, physical_multiplier = make_optimizer(model)
    scheduler, warmup_epochs, minimum_lr = make_scheduler(optimizer, peak_lr)
    scaler = torch.amp.GradScaler(AMP_DEVICE_TYPE, enabled=USE_AMP)

    hindcast_start = float(os.getenv("V112_HINDCAST_WEIGHT_START", "0.70"))
    hindcast_end = float(os.getenv("V112_HINDCAST_WEIGHT_END", "0.10"))
    hindcast_decay_epochs = int(os.getenv("V112_HINDCAST_DECAY_EPOCHS", "8"))
    transit_l2_weight = float(os.getenv("V112_TRANSIT_RESIDUAL_L2", "0.003"))
    alignment_weight = float(os.getenv("V112_ALIGNMENT_WEIGHT", "0.02"))
    alignment_sigma_deg = float(os.getenv("V112_ALIGNMENT_SIGMA_DEG", "20"))
    surge_weight = float(os.getenv("V112_SURGE_WEIGHT", "0.02"))
    surge_pos_weight = torch.tensor(
        float(os.getenv("V112_SURGE_POS_WEIGHT", "2.3")), device=DEVICE
    )
    patience = int(os.getenv("V112_EARLY_STOP_PATIENCE", "15"))

    manifest = {
        "train": chain_manifest(train_chains),
        "validation": chain_manifest(val_chains),
        "train_rows_used": int(len(selected_train_index)),
        "validation_rows_used": int(len(selected_val_index)),
        "consecutive_pairs": pair_counts,
        "source_model": "torazang65 V7 commit 2df99c8 via V11.1",
        "v11_2_changes": [
            "dynamic longitude grid",
            "source-weight-aware time masking",
            "exact base fallback on modality dropout",
            "optional consecutive-chain forecast consistency",
        ],
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
        f"wind_only={wind_only} image_size={IMAGE_SIZE} "
        f"source_grid=20x{model.grid_rows}x{model.grid_columns} "
        f"train_chains={train_chains.count} val_chains={val_chains.count} "
        f"chain_balanced={chain_balanced} consistency_weight={consistency_weight:.3f} "
        f"pairs={pair_counts['train']}/{pair_counts['validation']} "
        f"lr={peak_lr:.2e} physical_lr_mult={physical_multiplier:.1f} "
        f"mask={SOLAR_DISK_MASK} norm={IMAGE_NORM} "
        f"loss=seokho_v7_rmse+hindcast+backmapping+optional_consistency"
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
            transit_l2_weight,
            alignment_weight,
            alignment_sigma_deg,
            surge_weight,
            surge_pos_weight,
            consistency_weight,
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
                transit_l2_weight,
                alignment_weight,
                alignment_sigma_deg,
                surge_weight,
                surge_pos_weight,
                consistency_weight,
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
            "val_base_rmse_km_s": val_metrics["base_rmse"],
            "train_hindcast_rmse_km_s": train_metrics["hindcast_rmse"],
            "val_hindcast_rmse_km_s": val_metrics["hindcast_rmse"],
            "train_alignment_kl": train_metrics["alignment_kl"],
            "val_alignment_kl": val_metrics["alignment_kl"],
            "train_consistency_rmse_km_s": train_metrics["consistency_rmse"],
            "val_consistency_rmse_km_s": val_metrics["consistency_rmse"],
            "train_consistency_pairs": train_metrics["consistency_pairs"],
            "val_consistency_pairs": val_metrics["consistency_pairs"],
            "val_surge_auroc": val_metrics["surge_auroc"],
            "val_propagation_rms_km_s": val_metrics["propagation_rms"],
            "hindcast_weight": hindcast_weight,
            "consistency_weight": consistency_weight,
            "learning_rate": learning_rate,
            "seconds": elapsed,
            **{
                f"val_{name}": value
                for name, value in val_metrics["diagnostics"].items()
            },
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        consistency_text = (
            f" consistency={val_metrics['consistency_rmse']:.3f}"
            if math.isfinite(val_metrics["consistency_rmse"])
            else ""
        )
        print(
            f"epoch={epoch:03d} train_rmse={train_metrics['rmse']:.3f} "
            f"val_rmse={val_metrics['rmse']:.3f} "
            f"val_chain_macro_rmse={val_metrics['chain_macro_rmse']:.3f} "
            f"base_val_rmse={val_metrics['base_rmse']:.3f} "
            f"hind={val_metrics['hindcast_rmse']:.1f} "
            f"align_kl={val_metrics['alignment_kl']:.3f} "
            f"auroc={val_metrics['surge_auroc']:.3f}"
            f"{consistency_text} "
            f"lr={learning_rate:.2e} seconds={elapsed:.1f}",
            flush=True,
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "architecture": ARCHITECTURE_NAME,
                    "version": "11.2",
                    "model_state_dict": model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "epoch": epoch,
                    "val_rmse_km_s": val_metrics["rmse"],
                    "val_chain_macro_rmse_km_s": val_metrics[
                        "chain_macro_rmse"
                    ],
                    "val_consistency_rmse_km_s": val_metrics[
                        "consistency_rmse"
                    ],
                    "channels": CHANNELS,
                    "preprocess": {
                        "image_size": IMAGE_SIZE,
                        "image_norm": IMAGE_NORM,
                        "soft_cubic_strength": SOFT_CUBIC_STRENGTH,
                        "solar_disk_mask": SOLAR_DISK_MASK,
                        "solar_disk_radius_fraction": SOLAR_DISK_RADIUS_FRACTION,
                        "feature_schema": FEATURE_SCHEMA,
                        "source_grid": [
                            20,
                            model.grid_rows,
                            model.grid_columns,
                        ],
                        "warmup_epochs": warmup_epochs,
                        "minimum_learning_rate": minimum_lr,
                        "optimizer_weight_decay": weight_decay,
                        "alignment_sigma_deg": alignment_sigma_deg,
                        "consistency_weight": consistency_weight,
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
    print(f"saved: {checkpoint_path.resolve()} best_val_rmse={best_val_rmse:.3f}")


if __name__ == "__main__":
    main()
