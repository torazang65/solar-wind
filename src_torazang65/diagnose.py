"""학습된 체크포인트 하나를 받아 val/train을 분해 진단한다.

사용법:
    python diagnose.py                       # OUTPUT_DIR/best_model.pth
    python diagnose.py path/to/best_model.pth
    WIND_ONLY=1 python diagnose.py           # wind_only 런 진단

train.py는 건드리지 않는다. 리포트는 stdout으로 출력되고,
val 샘플별 오차 CSV가 체크포인트 옆에 저장된다.

리포트 번호는 진단 논의 때의 번호를 따른다:
  [1] 앵커 베이스라인 (persistence / climatology)
  [2] val 스파이크 범인 추적 (worst-k 샘플, 이벤트 기여도)
  [3] horizon별 분해 (모델 vs persistence)
  [4] 평균 회귀(이벤트 축소 예측) 여부
  [6] clean-train 평가 (증강 끈 진짜 train-val 간격)
  [7] train/val 구성 비교 (분포 이동 확인)
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

from config import *
from model import SolarWindBaseline
from dataset import (
    info,
    train_inputs,
    train_loader,
    train_targets,
    val_inputs,
    val_loader,
    val_targets,
)

WIND_ONLY = os.environ.get("WIND_ONLY") == "1"
RUN_DIR = OUTPUT_DIR / "wind_only" if WIND_ONLY else OUTPUT_DIR

checkpoint_path = (
    Path(sys.argv[1]) if len(sys.argv) > 1 else RUN_DIR / "best_model.pth"
)

# 이벤트 윈도우 판정 기준: 12개 horizon 중 하나라도 last_wind 대비
# 이만큼(km/s) 이상 변하면 이벤트로 본다.
EVENT_THRESHOLD_KM_S = 100.0
WORST_K = 15

HORIZON_HOURS = [(index + 1) * 6 for index in range(12)]


def rmse(error):
    """km/s 단위 오차 배열 -> 스칼라 RMSE."""
    return float(np.sqrt(np.mean(np.square(error))))


@torch.no_grad()
def collect(loader):
    """loader 전체를 eval 모드로 순회해 km/s 단위 배열을 모은다.

    반환: predictions (N,12), targets (N,12), last_wind (N,), sample_ids (N,)
    증강(time mask / modality drop)은 model.eval()이라 모두 꺼진 상태.
    """
    model.eval()
    predictions, targets, last_wind, sample_ids = [], [], [], []
    for batch in loader:
        images = batch["images"].to(DEVICE, non_blocking=PIN_MEMORY)
        wind = batch["wind"].to(DEVICE, non_blocking=PIN_MEMORY)
        prediction = model(images, wind)
        predictions.append(prediction.float().cpu().numpy())
        targets.append(batch["target"].numpy())
        last_wind.append(batch["wind"][:, -1].numpy())
        ids = batch["sample_id"]
        ids = ids.numpy() if torch.is_tensor(ids) else np.asarray(ids)
        sample_ids.append(ids)
    return (
        np.concatenate(predictions) * 1000.0,
        np.concatenate(targets) * 1000.0,
        np.concatenate(last_wind) * 1000.0,
        np.concatenate(sample_ids),
    )


def banner(title):
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


# ==========================================================
# 모델 로드
# ==========================================================
model = SolarWindBaseline(
    image_size=IMAGE_SIZE, use_images=not WIND_ONLY, **MODEL_KWARGS
).to(DEVICE)
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
print(f"checkpoint: {checkpoint_path}")
print(
    f"  epoch={checkpoint.get('epoch')} "
    f"val_rmse_km_s={checkpoint.get('val_rmse_km_s'):.3f} "
    f"wind_only={WIND_ONLY}"
)

print("collecting val predictions ...", flush=True)
val_pred, val_true, val_last, val_ids = collect(val_loader)
print("collecting clean-train predictions ...", flush=True)
train_pred, train_true, train_last, _ = collect(train_loader)

# persistence: 12개 horizon 전부 last_wind로 예측
val_persistence = np.repeat(val_last[:, None], 12, axis=1)
train_persistence = np.repeat(train_last[:, None], 12, axis=1)

# climatology: train target의 horizon별 평균 (train_targets는 raw km/s)
train_climatology = train_targets.mean(axis=0)
val_climatology_oracle = val_true.mean(axis=0)

# ==========================================================
# [1] 앵커 베이스라인
# ==========================================================
banner("[1] 앵커 베이스라인 (val 전체 RMSE, km/s)")
print(f"model                    : {rmse(val_pred - val_true):8.3f}")
print(f"persistence (wind[-1])   : {rmse(val_persistence - val_true):8.3f}")
print(f"climatology (train mean) : {rmse(train_climatology[None] - val_true):8.3f}")
# val 자체 평균으로 만든 climatology와의 차이는 train/val 분포 이동의 신호다.
print(f"climatology (val mean)*  : {rmse(val_climatology_oracle[None] - val_true):8.3f}")
print("  * val 평균은 실전에서 쓸 수 없는 oracle. train mean과의 격차가")
print("    크면 두 split의 수준 자체가 다르다는 뜻 ([7]과 같이 볼 것).")

# ==========================================================
# [3] horizon별 분해  +  [4]의 축소비율(shrinkage)
# ==========================================================
banner("[3] horizon별 RMSE 분해 (km/s)  /  shrinkage=std(pred-last)/std(true-last)")
val_delta_true = val_true - val_last[:, None]
val_delta_pred = val_pred - val_last[:, None]
print(f"{'horizon':>8} {'model':>9} {'persist':>9} {'clim(tr)':>9} {'shrink':>7}")
for index, hours in enumerate(HORIZON_HOURS):
    model_rmse = rmse(val_pred[:, index] - val_true[:, index])
    persistence_rmse = rmse(val_persistence[:, index] - val_true[:, index])
    climatology_rmse = rmse(train_climatology[index] - val_true[:, index])
    shrink = float(
        np.std(val_delta_pred[:, index]) / np.std(val_delta_true[:, index])
    )
    marker = "  <- model wins" if model_rmse < persistence_rmse else ""
    print(
        f"{hours:>6}h {model_rmse:>9.3f} {persistence_rmse:>9.3f} "
        f"{climatology_rmse:>9.3f} {shrink:>7.2f}{marker}"
    )

# ==========================================================
# [4] 평균 회귀 여부
# ==========================================================
banner("[4] 평균 회귀 여부: |pred-last| vs |true-last| 분포 (km/s, val 전체)")
absolute_pred = np.abs(val_delta_pred).ravel()
absolute_true = np.abs(val_delta_true).ravel()
print(f"{'':>12} {'p50':>8} {'p90':>8} {'p99':>8} {'max':>8} {'std':>8}")
for label, values in (("true", absolute_true), ("pred", absolute_pred)):
    q50, q90, q99 = np.percentile(values, [50, 90, 99])
    print(
        f"{label:>12} {q50:>8.1f} {q90:>8.1f} {q99:>8.1f} "
        f"{values.max():>8.1f} {values.std():>8.1f}"
    )
print("  pred의 분포가 true보다 크게 좁으면 이벤트를 포기하고 밋밋하게")
print("  내는 상태다. horizon별 정도는 [3]의 shrink 열 참고.")

# ==========================================================
# [2] 스파이크 범인 추적: 이벤트 분해 + worst-k
# ==========================================================
banner(f"[2] 이벤트/조용 분해 (이벤트: max|true-last| >= {EVENT_THRESHOLD_KM_S:.0f} km/s)")
delta_max = np.abs(val_delta_true).max(axis=1)
is_event = delta_max >= EVENT_THRESHOLD_KM_S
squared_error = np.square(val_pred - val_true)
event_share = squared_error[is_event].sum() / squared_error.sum()
for label, mask in (("event", is_event), ("quiet", ~is_event)):
    if mask.sum() == 0:
        print(f"{label:>6}: 0 샘플")
        continue
    print(
        f"{label:>6}: n={int(mask.sum()):5d} ({mask.mean():5.1%})  "
        f"model={rmse((val_pred - val_true)[mask]):8.3f}  "
        f"persistence={rmse((val_persistence - val_true)[mask]):8.3f}"
    )
print(f"전체 제곱오차 중 이벤트 샘플 기여: {event_share:.1%}")
print("  -> 이 비율이 크면 val 널뛰기는 소수 이벤트 윈도우가 만드는 것.")

per_sample = pd.DataFrame({
    "sample_id": val_ids,
    "rmse_model": np.sqrt(squared_error.mean(axis=1)),
    "rmse_persistence": np.sqrt(np.square(val_persistence - val_true).mean(axis=1)),
    "last_wind": val_last,
    "target_min": val_true.min(axis=1),
    "target_max": val_true.max(axis=1),
    "delta_max": delta_max,
    "is_event": is_event,
})
per_sample_path = checkpoint_path.parent / "diagnostics_per_sample.csv"
per_sample.sort_values("rmse_model", ascending=False).to_csv(
    per_sample_path, index=False
)

banner(f"[2] worst-{WORST_K} val 샘플 (이 체크포인트 기준)")
worst = per_sample.nlargest(WORST_K, "rmse_model")
print(worst.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
print(f"\n샘플별 전체 CSV 저장: {per_sample_path}")
print("  다른 체크포인트로 다시 돌려 worst 목록이 겹치는지 보면")
print("  '항상 같은 윈도우를 못 맞추는지'를 확인할 수 있다.")

# ==========================================================
# [6] clean-train 평가
# ==========================================================
banner("[6] clean-train 평가 (증강 끔, model.eval())")
clean_train_rmse = rmse(train_pred - train_true)
clean_val_rmse = rmse(val_pred - val_true)
print(f"train RMSE (clean)       : {clean_train_rmse:8.3f}")
print(f"val   RMSE               : {clean_val_rmse:8.3f}")
print(f"진짜 train-val 간격       : {clean_val_rmse - clean_train_rmse:8.3f}")
print(f"train persistence 참고    : {rmse(train_persistence - train_true):8.3f}")
print("  train.py 로그의 train_rmse는 증강이 켜진 값이라 이보다 크다.")

# ==========================================================
# [7] train/val 구성 비교
# ==========================================================
banner("[7] train/val 구성 비교 (분포 이동 확인)")
print(f"샘플 수: train={len(train_inputs)}  val={len(val_inputs)}")
print(f"{'':>22} {'train':>10} {'val':>10}")
rows = [
    ("target mean", train_targets.mean(), val_targets.mean()),
    ("target std", train_targets.std(), val_targets.std()),
    ("target p99", np.percentile(train_targets, 99), np.percentile(val_targets, 99)),
    ("last_wind mean", train_last.mean(), val_last.mean()),
    ("last_wind std", train_last.std(), val_last.std()),
]
for label, train_value, val_value in rows:
    print(f"{label:>22} {train_value:>10.1f} {val_value:>10.1f}")

for frame, name in ((train_inputs, "train"), (val_inputs, "val")):
    id_low, id_high = frame.sample_id.min(), frame.sample_id.max()
    file_low = frame.image_00.min()
    file_high = frame.image_00.max()
    print(f"{name:>6}: sample_id [{id_low} .. {id_high}]")
    print(f"{'':>6}  image_00  [{file_low} .. {file_high}]")
print("  파일명/sample_id에 날짜가 들어 있으면 위 범위로 두 split의")
print("  기간(태양활동 위상)이 갈리는지 읽을 수 있다.")

info_text = json.dumps(info, ensure_ascii=False)
print(f"\ndataset_info.json (앞 600자):\n{info_text[:600]}")
