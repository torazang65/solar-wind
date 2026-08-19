import time
import math
import torch
from torch.nn import functional as F
import pandas as pd
import matplotlib.pyplot as plt
from config import *
from model import SolarWindBaseline
from dataset import train_loader, val_loader

# 모델 및 옵티마이저 초기화
model = SolarWindBaseline(image_size=IMAGE_SIZE).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.25, patience=3, min_lr=1e-6
)
scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP)
checkpoint_path = OUTPUT_DIR / "best_model.pth"

if checkpoint_path.exists():
    checkpoint_path.unlink()

patience = 5
best_val_rmse = float("inf")
epochs_without_improvement = 0
history = []

def run_epoch(loader, training):
    model.train(training)
    squared_error_sum = 0.0
    value_count = 0
    for batch in loader:
        images = batch["images"].to(DEVICE, non_blocking=PIN_MEMORY)
        wind = batch["wind"].to(DEVICE, non_blocking=PIN_MEMORY)
        target = batch["target"].to(DEVICE, non_blocking=PIN_MEMORY)
        
        if training:
            optimizer.zero_grad(set_to_none=True)
            
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(DEVICE.type, enabled=USE_AMP):
                prediction = model(images, wind)
                mse = F.mse_loss(prediction, target)
                loss = torch.sqrt(mse + RMSE_EPSILON)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
        error_km_s = (prediction.detach() - target) * 1000.0
        squared_error_sum += float(torch.sum(error_km_s ** 2).cpu())
        value_count += error_km_s.numel()
        
    return math.sqrt(squared_error_sum / value_count)

if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        train_rmse = run_epoch(train_loader, training=True)
        with torch.no_grad():
            val_rmse = run_epoch(val_loader, training=False)
            
        scheduler.step(val_rmse)
        learning_rate = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - started
        
        history.append({
            "epoch": epoch, "train_rmse_km_s": train_rmse,
            "val_rmse_km_s": val_rmse, "learning_rate": learning_rate, "seconds": elapsed,
        })
        print(f"epoch={epoch:03d} train_rmse={train_rmse:.3f} val_rmse={val_rmse:.3f} lr={learning_rate:.2e} seconds={elapsed:.1f}")
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_rmse_km_s": val_rmse,
                "channels": CHANNELS,
            }, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("early stopping")
                break

    # 시각화 및 저장
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(OUTPUT_DIR / "history.csv", index=False)
    history_frame.plot(x="epoch", y=["train_rmse_km_s", "val_rmse_km_s"], grid=True)
    plt.ylabel("RMSE (km/s)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "learning_curve.png", dpi=140)
    print("Training finished.")