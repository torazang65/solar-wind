import os
import time
import math
import torch
from torch.nn import functional as F
import pandas as pd
import matplotlib.pyplot as plt
from config import *
from model import SolarWindBaseline
from dataset import train_loader, val_loader

# WIND_ONLY=1 로 실행하면 CNN을 건너뛰고 wind 히스토리만으로 학습한다.
# 이미지 경로가 일반화에 실제로 기여하는지 확인하는 진단용 실행이며,
# 결과는 별도 디렉토리에 저장되어 본 실행의 산출물을 덮어쓰지 않는다.
WIND_ONLY = os.environ.get("WIND_ONLY") == "1"
RUN_DIR = OUTPUT_DIR / "wind_only" if WIND_ONLY else OUTPUT_DIR
RUN_DIR.mkdir(parents=True, exist_ok=True)

if WIND_ONLY:
    print("=" * 60)
    print("WIND-ONLY 진단 실행: 이미지 입력을 사용하지 않습니다")
    print(f"출력 경로: {RUN_DIR}")
    print("=" * 60, flush=True)

# 모델 및 옵티마이저 초기화
model = SolarWindBaseline(
    image_size=IMAGE_SIZE, use_images=not WIND_ONLY, **MODEL_KWARGS
).to(DEVICE)
print(f"파라미터 수: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=0.01)

# warmup 후 cosine, epoch 단위 step. LambdaLR 인자는 지금까지 step된
# 횟수(0-index)라 첫 에폭은 PEAK_LR * 1/WARMUP_EPOCHS에서 시작하고,
# EPOCHS 도달 시점에 MIN_LR 근처까지 내려간다.
def lr_lambda(step):
    if step < WARMUP_EPOCHS:
        return (step + 1) / WARMUP_EPOCHS
    progress = (step - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    floor = MIN_LR / PEAK_LR
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP)
checkpoint_path = RUN_DIR / "best_model.pth"

if checkpoint_path.exists():
    checkpoint_path.unlink()

# cosine은 개선이 후반 tail에서 잘게 오는데 val이 이벤트 주도로 ±10씩
# 널뛰므로, 12로는 스케줄이 끝나기 전에 끊길 위험이 있어 여유를 둔다.
# 에폭당 ~17초라 늘려도 비용은 미미하다.
patience = 20
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
            
        # 이번 에폭이 실제로 쓴 lr을 기록한 뒤 다음 에폭 값으로 step.
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
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
    history_frame.to_csv(RUN_DIR / "history.csv", index=False)
    history_frame.plot(x="epoch", y=["train_rmse_km_s", "val_rmse_km_s"], grid=True)
    plt.ylabel("RMSE (km/s)")
    plt.tight_layout()
    plt.savefig(RUN_DIR / "learning_curve.png", dpi=140)
    print(f"Training finished. best_val_rmse={best_val_rmse:.3f}")