import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from config import *
from dataset import (
    WIND_COLUMNS,
    train_inputs,
    train_loader,
    train_targets,
    val_inputs,
    val_loader,
    val_targets,
)
from model_transformer import SolarWindTransformer


def fit_horizon_baseline(inputs, targets):
    last_wind = inputs[WIND_COLUMNS[-1]].to_numpy(np.float32) / 1000.0
    target = np.asarray(targets, dtype=np.float32) / 1000.0

    centered_wind = last_wind - last_wind.mean()
    centered_target = target - target.mean(axis=0, keepdims=True)
    denominator = float(np.sum(centered_wind ** 2))
    slope = np.sum(centered_wind[:, None] * centered_target, axis=0) / denominator
    intercept = target.mean(axis=0) - slope * last_wind.mean()
    return slope.astype(np.float32), intercept.astype(np.float32)


WIND_ONLY = os.getenv("WIND_ONLY", "0").lower() in {"1", "true", "yes"}
baseline_slope, baseline_intercept = fit_horizon_baseline(train_inputs, train_targets)

model_kwargs = {
    "image_size": IMAGE_SIZE,
    "apply_solar_disk_mask": SOLAR_DISK_MASK,
    "solar_disk_center_fraction": SOLAR_DISK_CENTER_FRACTION,
    "solar_disk_radius_fraction": SOLAR_DISK_RADIUS_FRACTION,
    "spatial_feature_size": SPATIAL_FEATURE_SIZE,
    "use_images": not WIND_ONLY,
    **TRANSFORMER_KWARGS,
}
model = SolarWindTransformer(
    baseline_slope=baseline_slope,
    baseline_intercept=baseline_intercept,
    **model_kwargs,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.25, patience=3, min_lr=1e-6
)
scaler = torch.amp.GradScaler(AMP_DEVICE_TYPE, enabled=USE_AMP)
checkpoint_path = OUTPUT_DIR / "best_transformer.pth"

if checkpoint_path.exists():
    checkpoint_path.unlink()

patience = 8
best_val_rmse = float("inf")
epochs_without_improvement = 0
history = []


def baseline_validation_rmse():
    last_wind = val_inputs[WIND_COLUMNS[-1]].to_numpy(np.float32) / 1000.0
    target = np.asarray(val_targets, dtype=np.float32) / 1000.0
    prediction = last_wind[:, None] * baseline_slope + baseline_intercept
    return float(np.sqrt(np.mean((prediction - target) ** 2)) * 1000.0)


def run_epoch(loader, training):
    model.train(training)
    squared_error_sum = 0.0
    value_count = 0

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["images"].to(DEVICE, non_blocking=PIN_MEMORY)
        wind = batch["wind"].to(DEVICE, non_blocking=PIN_MEMORY)
        target = batch["target"].to(DEVICE, non_blocking=PIN_MEMORY)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(AMP_DEVICE_TYPE, enabled=USE_AMP):
                prediction = model(images, wind)
                loss = F.mse_loss(prediction, target)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        error_km_s = (prediction.detach() - target) * 1000.0
        squared_error_sum += float(torch.sum(error_km_s ** 2).cpu())
        value_count += error_km_s.numel()

        if batch_index % 20 == 0 or batch_index == len(loader):
            running_rmse = math.sqrt(squared_error_sum / value_count)
            mode = "train" if training else "val"
            print(
                f"{mode} batch={batch_index}/{len(loader)} "
                f"running_rmse={running_rmse:.3f}",
                flush=True,
            )

    return math.sqrt(squared_error_sum / value_count)


if __name__ == "__main__":
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"device={DEVICE} parameters={parameter_count:,} wind_only={WIND_ONLY}")
    print(f"linear_baseline_val_rmse={baseline_validation_rmse():.3f}")
    print(f"baseline_slope={np.round(baseline_slope, 3).tolist()}")

    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        train_rmse = run_epoch(train_loader, training=True)
        with torch.no_grad():
            val_rmse = run_epoch(val_loader, training=False)

        scheduler.step(val_rmse)
        learning_rate = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - started
        history.append(
            {
                "epoch": epoch,
                "train_rmse_km_s": train_rmse,
                "val_rmse_km_s": val_rmse,
                "learning_rate": learning_rate,
                "seconds": elapsed,
            }
        )
        print(
            f"epoch={epoch:03d} train_rmse={train_rmse:.3f} "
            f"val_rmse={val_rmse:.3f} lr={learning_rate:.2e} "
            f"seconds={elapsed:.1f}",
            flush=True,
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "epoch": epoch,
                    "val_rmse_km_s": val_rmse,
                    "channels": CHANNELS,
                    "preprocess": {
                        "image_norm": IMAGE_NORM,
                        "soft_cubic_strength": SOFT_CUBIC_STRENGTH,
                        "solar_disk_mask": SOLAR_DISK_MASK,
                        "solar_disk_radius_fraction": SOLAR_DISK_RADIUS_FRACTION,
                    },
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("early stopping")
                break

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(OUTPUT_DIR / "transformer_history.csv", index=False)
    history_frame.plot(
        x="epoch", y=["train_rmse_km_s", "val_rmse_km_s"], grid=True
    )
    plt.ylabel("RMSE (km/s)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "transformer_learning_curve.png", dpi=140)
    print(f"saved: {checkpoint_path.resolve()}")
