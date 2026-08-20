import time

import matplotlib.pyplot as plt
import pandas as pd
import torch

import train_solar_geometry_v3 as training
from config import *
from model_solar_cartesian_v4 import SolarWindCartesianTransformerV4


model_kwargs = dict(training.model_kwargs)
model = SolarWindCartesianTransformerV4(
    baseline_slope=training.baseline_slope,
    baseline_intercept=training.baseline_intercept,
    **model_kwargs,
).to(DEVICE)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=SOLAR_PROBABILISTIC_LR, weight_decay=0.03
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.25, patience=3, min_lr=1e-6
)
scaler = torch.amp.GradScaler(AMP_DEVICE_TYPE, enabled=USE_AMP)

# Reuse the tested V3 epoch runner with the Cartesian model and optimizer.
training.model = model
training.optimizer = optimizer
training.scheduler = scheduler
training.scaler = scaler

checkpoint_path = OUTPUT_DIR / "best_solar_cartesian_v4.pth"
history_path = OUTPUT_DIR / "solar_cartesian_v4_history.csv"
if checkpoint_path.exists():
    checkpoint_path.unlink()


if __name__ == "__main__":
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"architecture=solar_cartesian_v4 device={DEVICE} "
        f"parameters={parameter_count:,} wind_only={training.WIND_ONLY} "
        f"image_size={IMAGE_SIZE} "
        f"spatial_grid={training.SPATIAL_HEIGHT}x{training.SPATIAL_WIDTH} "
        f"disk_radius={SOLAR_DISK_RADIUS_FRACTION:.3f} "
        f"image_norm={IMAGE_NORM} lr={SOLAR_PROBABILISTIC_LR:.2e} "
        f"dropout={model_kwargs['dropout']:.2f} "
        f"visual_dropout={training.VISUAL_DROPOUT:.2f} loss=mse_km_s"
    )
    print(f"linear_baseline_val_rmse={training.baseline_validation_rmse():.3f}")

    patience = 8
    best_val_rmse = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        train_rmse, train_residual_rms = training.run_epoch(
            training.train_loader, training=True
        )
        with torch.no_grad():
            val_rmse, val_residual_rms = training.run_epoch(
                training.val_loader, training=False
            )

        scheduler.step(val_rmse)
        learning_rate = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - started
        history.append(
            {
                "epoch": epoch,
                "train_rmse_km_s": train_rmse,
                "val_rmse_km_s": val_rmse,
                "train_residual_rms_km_s": train_residual_rms,
                "val_residual_rms_km_s": val_residual_rms,
                "learning_rate": learning_rate,
                "seconds": elapsed,
            }
        )
        pd.DataFrame(history).to_csv(history_path, index=False)
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
                    "architecture": "SolarWindCartesianTransformerV4",
                    "version": 4,
                    "model_state_dict": model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "epoch": epoch,
                    "val_rmse_km_s": val_rmse,
                    "channels": CHANNELS,
                    "preprocess": {
                        "image_size": IMAGE_SIZE,
                        "image_norm": IMAGE_NORM,
                        "soft_cubic_strength": SOFT_CUBIC_STRENGTH,
                        "solar_disk_mask": SOLAR_DISK_MASK,
                        "solar_disk_radius_fraction": SOLAR_DISK_RADIUS_FRACTION,
                        "geometry": "cartesian_disk",
                        "darkness": "relative_deficit_sqrt_mu",
                        "spatial_grid": [
                            training.SPATIAL_HEIGHT,
                            training.SPATIAL_WIDTH,
                        ],
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
    history_frame.plot(
        x="epoch", y=["train_rmse_km_s", "val_rmse_km_s"], grid=True
    )
    plt.ylabel("RMSE (km/s)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "solar_cartesian_v4_learning_curve.png", dpi=140)
    print(f"saved: {checkpoint_path.resolve()} best_val_rmse={best_val_rmse:.3f}")
