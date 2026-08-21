import gc
import os

import numpy as np
import pandas as pd
import torch

from chain_sampling import (
    ChainAwareSolarWindDataset,
    infer_temporal_chains,
    make_chain_loader,
)
from config import *
from dataset import (
    IMAGE_COLUMNS,
    TARGET_COLUMNS,
    SolarWindDataset,
    make_loader,
    test_image_array,
    test_image_index,
    test_index,
    test_inputs,
    val_image_array,
    val_image_index,
    val_index,
    val_inputs,
    val_targets,
)
from model_solar_lstm_v12 import (
    ARCHITECTURE_NAME,
    FILE_STEM,
    SolarWindLagLSTMV12,
)
from train_solar_lstm_v12 import FEATURE_SCHEMA, parse_lag_hours


MODEL_CLASS = SolarWindLagLSTMV12
CHECKPOINT_VERSION = "12"
EXTRA_PREPROCESS = {}


def current_preprocess():
    return {
        "image_size": IMAGE_SIZE,
        "image_norm": IMAGE_NORM,
        "soft_cubic_strength": SOFT_CUBIC_STRENGTH,
        "solar_disk_mask": SOLAR_DISK_MASK,
        "solar_disk_radius_fraction": SOLAR_DISK_RADIUS_FRACTION,
        "feature_schema": FEATURE_SCHEMA,
        "spatial_grid": [
            int(os.getenv("V12_GRID_ROWS", "2")),
            int(os.getenv("V12_GRID_COLUMNS", "8")),
        ],
        "lag_hours": list(parse_lag_hours()),
        "lag_prior_max_strength": float(
            os.getenv("V12_LAG_PRIOR_MAX_STRENGTH", "2.0")
        ),
        "lag_alignment_weight": float(
            os.getenv("V12_LAG_ALIGNMENT_WEIGHT", "0.01")
        ),
        **EXTRA_PREPROCESS,
    }


def load_best_model():
    checkpoint_path = OUTPUT_DIR / f"best_{FILE_STEM}.pth"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    if checkpoint.get("architecture") != ARCHITECTURE_NAME:
        raise ValueError(
            f"not a {ARCHITECTURE_NAME} checkpoint: "
            f"{checkpoint.get('architecture')}"
        )
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version: {checkpoint.get('version')}")
    expected_preprocess = checkpoint["preprocess"]
    for key, current in current_preprocess().items():
        expected = expected_preprocess[key]
        if current != expected:
            raise ValueError(
                "inference preprocessing does not match checkpoint: "
                f"key={key}, current={current}, checkpoint={expected}"
            )

    model = MODEL_CLASS(**checkpoint["model_kwargs"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    print(
        f"loaded {FILE_STEM} best epoch={checkpoint['epoch']} "
        f"val_rmse={checkpoint['val_rmse_km_s']:.3f} "
        f"val_chain_macro_rmse={checkpoint['val_chain_macro_rmse_km_s']:.3f} "
        f"image_size={model.image_size} "
        f"grid={model.grid_rows}x{model.grid_columns} "
        f"lag_hours={model.lag_hours.tolist()}"
    )
    return model, checkpoint


@torch.no_grad()
def predict(model, loader, return_components=False):
    predictions = []
    sample_ids = []
    chain_ids = []
    chain_positions = []
    component_values = {}
    for batch_index, batch in enumerate(loader, start=1):
        images = batch["images"].to(DEVICE, non_blocking=PIN_MEMORY)
        wind = batch["wind"].to(DEVICE, non_blocking=PIN_MEMORY)
        with torch.amp.autocast(AMP_DEVICE_TYPE, enabled=USE_AMP):
            if return_components:
                prediction, components = model(
                    images, wind, return_components=True
                )
            else:
                prediction = model(images, wind)
                components = {}
        predictions.append(prediction.float().cpu().numpy() * 1000.0)
        for name, value in components.items():
            scale = 1.0 if name == "correction_gate" else 1000.0
            component_values.setdefault(name, []).append(
                value.float().cpu().numpy() * scale
            )
        sample_ids.extend(batch["sample_id"])
        if "chain_id" in batch:
            chain_ids.extend(batch["chain_id"].numpy().tolist())
            chain_positions.extend(batch["chain_position"].numpy().tolist())
        if batch_index % 20 == 0 or batch_index == len(loader):
            print(f"inference batch={batch_index}/{len(loader)}", flush=True)
    return (
        np.concatenate(predictions),
        sample_ids,
        np.asarray(chain_ids),
        np.asarray(chain_positions),
        {
            name: np.concatenate(values)
            for name, values in component_values.items()
        },
    )


def metrics_by_horizon(actual, prediction):
    rows = []
    for index in range(12):
        error = prediction[:, index] - actual[:, index]
        denominator = np.std(actual[:, index]) * np.std(prediction[:, index])
        correlation = (
            float(np.corrcoef(actual[:, index], prediction[:, index])[0, 1])
            if denominator > 0.0
            else float("nan")
        )
        rows.append(
            {
                "horizon_hours": (index + 1) * 6,
                "rmse_km_s": float(np.sqrt(np.mean(error**2))),
                "mae_km_s": float(np.mean(np.abs(error))),
                "bias_km_s": float(np.mean(error)),
                "correlation": correlation,
            }
        )
    return pd.DataFrame(rows)


def main():
    model, _ = load_best_model()
    val_chains = infer_temporal_chains(val_inputs, IMAGE_COLUMNS)
    val_dataset = ChainAwareSolarWindDataset(
        val_image_array,
        val_image_index,
        val_inputs,
        val_index,
        val_targets,
        temporal_chains=val_chains,
    )
    val_loader = make_chain_loader(val_dataset, training=False)

    print(f"\n[Running {FILE_STEM} Validation Evaluation]")
    val_prediction, val_ids, chain_ids, chain_positions, components = predict(
        model, val_loader, return_components=True
    )
    val_actual = np.asarray(val_targets[val_index], dtype=np.float64)
    metrics_by_horizon(val_actual, val_prediction).to_csv(
        OUTPUT_DIR / f"{FILE_STEM}_validation_metrics.csv", index=False
    )
    prediction_frame = pd.DataFrame(
        val_prediction,
        columns=[f"prediction_{column}" for column in TARGET_COLUMNS],
    )
    prediction_frame.insert(0, "chain_position", chain_positions)
    prediction_frame.insert(0, "chain_id", chain_ids)
    prediction_frame.insert(0, "sample_id", val_ids)
    for name in ("wind_base", "image_correction", "correction_gate"):
        values = components[name]
        for index in range(values.shape[1]):
            prediction_frame[f"{name}_{index:02d}"] = values[:, index]
    actual_frame = pd.DataFrame(
        val_actual, columns=[f"actual_{column}" for column in TARGET_COLUMNS]
    )
    pd.concat([prediction_frame, actual_frame], axis=1).to_csv(
        OUTPUT_DIR / f"{FILE_STEM}_validation_predictions.csv", index=False
    )
    squared_error = (val_prediction - val_actual) ** 2
    overall_rmse = float(np.sqrt(np.mean(squared_error)))
    chain_macro_rmse = float(
        np.mean(
            [
                np.sqrt(np.mean(squared_error[chain_ids == chain_id]))
                for chain_id in np.unique(chain_ids)
            ]
        )
    )
    print(
        f"Overall validation RMSE: {overall_rmse:.3f} "
        f"chain macro RMSE: {chain_macro_rmse:.3f}"
    )

    del val_loader
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()

    print(f"\n[Running {FILE_STEM} Test Inference]")
    test_dataset = SolarWindDataset(
        test_image_array,
        test_image_index,
        test_inputs,
        test_index,
        targets=None,
    )
    test_loader = make_loader(test_dataset, shuffle=False)
    test_prediction, predicted_ids, _, _, _ = predict(model, test_loader)
    submission = pd.DataFrame(test_prediction, columns=TARGET_COLUMNS)
    submission.insert(0, "sample_id", predicted_ids)
    if submission.shape != (3868, 13):
        raise RuntimeError(f"unexpected submission shape: {submission.shape}")
    if submission.isna().any().any():
        raise RuntimeError("submission contains NaN values")
    submission_path = OUTPUT_DIR / f"{FILE_STEM}_submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"saved: {submission_path.resolve()}")
    print(f"shape: {submission.shape}")


if __name__ == "__main__":
    main()
