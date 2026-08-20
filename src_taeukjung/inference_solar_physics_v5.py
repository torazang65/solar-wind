import gc

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
from model_solar_physics_v5 import SolarWindPhysicsTransformerV5


def load_best_model(
    model_class=SolarWindPhysicsTransformerV5,
    architecture_name="SolarWindPhysicsTransformerV5",
    file_stem="solar_physics_v5",
    grid_label="cea_grid",
):
    checkpoint_path = OUTPUT_DIR / f"best_{file_stem}.pth"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    if checkpoint.get("architecture") != architecture_name:
        raise ValueError(
            f"not a {architecture_name} checkpoint: {checkpoint.get('architecture')}"
        )

    checkpoint_preprocess = checkpoint["preprocess"]
    current_preprocess = {
        "image_size": IMAGE_SIZE,
        "image_norm": IMAGE_NORM,
        "solar_disk_mask": SOLAR_DISK_MASK,
        "solar_disk_radius_fraction": SOLAR_DISK_RADIUS_FRACTION,
        "solar_cea_radius_fraction": SOLAR_CEA_RADIUS_FRACTION,
    }
    if (
        IMAGE_NORM == "soft_cubic"
        or checkpoint_preprocess["image_norm"] == "soft_cubic"
    ):
        current_preprocess["soft_cubic_strength"] = SOFT_CUBIC_STRENGTH
    for key, value in current_preprocess.items():
        if checkpoint_preprocess[key] != value:
            raise ValueError(
                "inference preprocessing does not match checkpoint: "
                f"key={key}, current={value}, checkpoint={checkpoint_preprocess[key]}"
            )

    model = model_class(**checkpoint["model_kwargs"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(
        f"loaded {file_stem} best epoch={checkpoint['epoch']} "
        f"val_rmse={checkpoint['val_rmse_km_s']:.3f} "
        f"val_chain_macro_rmse={checkpoint['val_chain_macro_rmse_km_s']:.3f} "
        f"image_size={checkpoint['model_kwargs']['image_size']} "
        f"{grid_label}={checkpoint['model_kwargs']['latitude_bins']}x"
        f"{checkpoint['model_kwargs']['longitude_bins']}"
    )
    return model


@torch.no_grad()
def predict(model, loader):
    predictions, sample_ids, chain_ids = [], [], []
    for batch_index, batch in enumerate(loader, start=1):
        images = batch["images"].to(DEVICE, non_blocking=PIN_MEMORY)
        wind = batch["wind"].to(DEVICE, non_blocking=PIN_MEMORY)
        with torch.amp.autocast(AMP_DEVICE_TYPE, enabled=USE_AMP):
            prediction = model(images, wind)
        predictions.append(prediction.float().cpu().numpy() * 1000.0)
        sample_ids.extend(batch["sample_id"])
        if "chain_id" in batch:
            chain_ids.extend(batch["chain_id"].numpy().tolist())
        if batch_index % 20 == 0 or batch_index == len(loader):
            print(f"inference batch={batch_index}/{len(loader)}", flush=True)
    return np.concatenate(predictions), sample_ids, np.asarray(chain_ids)


def metrics_by_horizon(y_true, y_pred):
    rows = []
    for index in range(12):
        actual = y_true[:, index]
        predicted = y_pred[:, index]
        error = predicted - actual
        denominator = np.std(actual) * np.std(predicted)
        correlation = (
            float(np.corrcoef(actual, predicted)[0, 1])
            if denominator > 0
            else np.nan
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


def main(
    model_class=SolarWindPhysicsTransformerV5,
    architecture_name="SolarWindPhysicsTransformerV5",
    file_stem="solar_physics_v5",
    grid_label="cea_grid",
):
    model = load_best_model(
        model_class, architecture_name, file_stem, grid_label=grid_label
    )
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

    print(f"\n[Running {file_stem} Validation Evaluation]")
    validation_prediction, validation_ids, chain_ids = predict(model, val_loader)
    validation_target = np.asarray(val_targets[val_index], dtype=np.float64)
    validation_metrics = metrics_by_horizon(
        validation_target, validation_prediction
    )
    validation_metrics.to_csv(
        OUTPUT_DIR / f"{file_stem}_validation_metrics.csv", index=False
    )
    validation_frame = pd.DataFrame(
        validation_prediction,
        columns=[f"prediction_{column}" for column in TARGET_COLUMNS],
    )
    validation_frame.insert(0, "chain_id", chain_ids)
    validation_frame.insert(0, "sample_id", validation_ids)
    actual_frame = pd.DataFrame(
        validation_target,
        columns=[f"actual_{column}" for column in TARGET_COLUMNS],
    )
    pd.concat([validation_frame, actual_frame], axis=1).to_csv(
        OUTPUT_DIR / f"{file_stem}_validation_predictions.csv", index=False
    )
    squared_error = (validation_prediction - validation_target) ** 2
    overall_rmse = float(np.sqrt(np.mean(squared_error)))
    chain_rmse = [
        float(np.sqrt(np.mean(squared_error[chain_ids == chain_id])))
        for chain_id in np.unique(chain_ids)
    ]
    print(
        f"Overall validation RMSE: {overall_rmse:.3f} "
        f"chain macro RMSE: {np.mean(chain_rmse):.3f}"
    )

    del val_loader
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()

    print(f"\n[Running {file_stem} Test Inference]")
    test_dataset = SolarWindDataset(
        test_image_array,
        test_image_index,
        test_inputs,
        test_index,
        targets=None,
    )
    test_loader = make_loader(test_dataset, shuffle=False)
    test_prediction, predicted_ids, _ = predict(model, test_loader)
    submission = pd.DataFrame(test_prediction, columns=TARGET_COLUMNS)
    submission.insert(0, "sample_id", predicted_ids)
    submission_path = OUTPUT_DIR / f"{file_stem}_submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"saved: {submission_path.resolve()}")
    print(f"shape: {submission.shape}")


if __name__ == "__main__":
    main()
