import gc

import numpy as np
import pandas as pd
import torch

from config import *
from dataset import (
    TARGET_COLUMNS,
    WIND_COLUMNS,
    SolarWindDataset,
    make_loader,
    test_image_array,
    test_image_index,
    test_index,
    test_inputs,
    val_index,
    val_inputs,
    val_loader,
    val_targets,
)
from model_solar_probabilistic_v2 import SolarWindProbabilisticTransformerV2
from probabilistic import marginal_standard_deviation


def load_best_model():
    checkpoint_path = OUTPUT_DIR / "best_solar_probabilistic_v2.pth"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    if checkpoint.get("architecture") != "SolarWindProbabilisticTransformerV2":
        raise ValueError(f"not a v2 checkpoint: {checkpoint.get('architecture')}")

    checkpoint_preprocess = checkpoint["preprocess"]
    current_preprocess = {
        "image_size": IMAGE_SIZE,
        "image_norm": IMAGE_NORM,
        "soft_cubic_strength": SOFT_CUBIC_STRENGTH,
        "solar_disk_mask": SOLAR_DISK_MASK,
        "solar_disk_radius_fraction": SOLAR_DISK_RADIUS_FRACTION,
    }
    for key, value in current_preprocess.items():
        if checkpoint_preprocess[key] != value:
            raise ValueError(
                "inference preprocessing does not match checkpoint: "
                f"key={key}, current={value}, checkpoint={checkpoint_preprocess[key]}"
            )

    model = SolarWindProbabilisticTransformerV2(
        **checkpoint["model_kwargs"]
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"loaded v2 best epoch={checkpoint['epoch']} "
        f"val_rmse={checkpoint['val_rmse_km_s']:.3f} "
        f"image_size={checkpoint['model_kwargs']['image_size']} "
        f"spatial_grid={checkpoint['model_kwargs']['spatial_height']}x"
        f"{checkpoint['model_kwargs']['spatial_width']}"
    )
    return model


@torch.no_grad()
def predict(model, loader):
    model.eval()
    predictions, standard_deviations, sample_ids = [], [], []
    for batch_index, batch in enumerate(loader, start=1):
        images = batch["images"].to(DEVICE, non_blocking=PIN_MEMORY)
        wind = batch["wind"].to(DEVICE, non_blocking=PIN_MEMORY)
        with torch.amp.autocast(AMP_DEVICE_TYPE, enabled=USE_AMP):
            prediction, diagonal_scale, factors, degrees_of_freedom = model(
                images, wind, return_distribution=True
            )
        standard_deviation = marginal_standard_deviation(
            diagonal_scale.float(), factors.float(), degrees_of_freedom.float()
        )
        predictions.append(prediction.float().cpu().numpy() * 1000.0)
        standard_deviations.append(standard_deviation.cpu().numpy())
        sample_ids.extend(batch["sample_id"])
        if batch_index % 20 == 0 or batch_index == len(loader):
            print(f"inference batch={batch_index}/{len(loader)}", flush=True)
    return (
        np.concatenate(predictions),
        np.concatenate(standard_deviations),
        sample_ids,
    )


def metrics_by_horizon(y_true, y_pred, predicted_std):
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
                "correlation": correlation,
                "predicted_std_km_s": float(np.mean(predicted_std[:, index])),
                "one_std_coverage": float(
                    np.mean(np.abs(error) <= predicted_std[:, index])
                ),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    model = load_best_model()

    print("\n[Running V2 Validation Evaluation]")
    validation_prediction, validation_std, _ = predict(model, val_loader)
    validation_target = np.asarray(val_targets[val_index], dtype=np.float64)
    validation_metrics = metrics_by_horizon(
        validation_target, validation_prediction, validation_std
    )
    last_wind = val_inputs.iloc[val_index][WIND_COLUMNS[-1]].to_numpy(np.float64)
    slope = model.baseline_slope.detach().cpu().numpy().reshape(-1)
    intercept = model.baseline_intercept.detach().cpu().numpy().reshape(-1) * 1000.0
    linear_baseline = last_wind[:, None] * slope + intercept
    baseline_rmse = np.sqrt(
        np.mean((linear_baseline - validation_target) ** 2, axis=0)
    )
    validation_metrics["linear_baseline_rmse_km_s"] = baseline_rmse
    validation_metrics.to_csv(
        OUTPUT_DIR / "solar_probabilistic_v2_validation_metrics.csv", index=False
    )
    overall_rmse = np.sqrt(np.mean((validation_prediction - validation_target) ** 2))
    print(f"Overall validation RMSE: {overall_rmse:.3f}")

    del val_loader
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()

    print("\n[Running V2 Test Inference]")
    test_dataset = SolarWindDataset(
        test_image_array,
        test_image_index,
        test_inputs,
        test_index,
        targets=None,
    )
    test_loader = make_loader(test_dataset, shuffle=False)
    test_prediction, test_std, predicted_ids = predict(model, test_loader)

    submission = pd.DataFrame(test_prediction, columns=TARGET_COLUMNS)
    submission.insert(0, "sample_id", predicted_ids)
    submission_path = OUTPUT_DIR / "solar_probabilistic_v2_submission.csv"
    submission.to_csv(submission_path, index=False)

    uncertainty = pd.DataFrame(test_std, columns=TARGET_COLUMNS)
    uncertainty.insert(0, "sample_id", predicted_ids)
    uncertainty_path = OUTPUT_DIR / "solar_probabilistic_v2_test_std.csv"
    uncertainty.to_csv(uncertainty_path, index=False)
    print(f"saved: {submission_path.resolve()}")
    print(f"saved: {uncertainty_path.resolve()}")
    print(f"shape: {submission.shape}")
