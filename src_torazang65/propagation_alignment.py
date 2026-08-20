"""v6a propagation-alignment: attention COM 진단의 kernel-weight 재적용.

v6a(use_correction=False)에는 decoder cross-attention이 없어
attention_analysis가 대상을 잃는다 (스크립트는 가드로 종료). 그러나 그
분석이 답하던 질문 -- "모델이 horizon마다/레짐마다 다른 시점의 이미지를
읽는가" -- 는 v6a 판정에서도 핵심이다. v6a에서 그 역할은 propagation
branch의 soft assignment

    w_t(h) = p_t * exp(-(h - a_t)^2 / (2 sigma^2))

가 수행하므로, horizon별 이미지 COM(hours ago)을

    COM(h) = sum_{b,t} w_bt(h) * age_t / sum_{b,t} w_bt(h)

로 계산해 ballistic 이론 곡선(transit(v_group) - h)과 비교한다.

decoder attention과의 결정적 차이: kernel COM은 구조상 horizon을 따라
움직일 수밖에 없다 (h가 kernel 중심을 훑는다). 따라서 판정 대상은
"움직이는가"(decoder는 이것부터 실패, ~55h 고정)가 아니라
  (1) 기울기: COM이 h를 따라 이론처럼 얕아지는가 (d COM / d h ~ -1)
  (2) 레짐 분화: slow가 fast보다 깊은 과거를 읽는가 (transit 차이)
다.

입력: <run_dir>/branch_decomposition/components.npz -- gate, arrival,
speed만 쓰므로 모델/원본 데이터 없이 로컬에서 돈다 (원격 학습 / 로컬
분석 워크플로). kernel_sigma_hours / dist_eff_h / fallback_weight가
npz에 있으면 사용하고(신형 branch_decomposition dump), 없으면 v5a
기본값으로 대체한다 (구형 npz -- coverage 열만 빠진다).

사용법:

    python propagation_alignment.py                    # config OUTPUT_DIR
    python propagation_alignment.py <run_dir>          # 임의 런 디렉토리

결과물: <run_dir>/propagation_alignment/ 아래
    summary.csv        horizon별 전체 COM + 이론 밴드 (+coverage)
    speed_summary.csv  속도 3분위별 COM vs 이론 COM
    propagation_com.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP_HOURS = 6.0
N_HORIZON = 12
SPEED_GROUP_LABELS = ("slow", "mid", "fast")
# 구형 components.npz(스칼라 미포함)용 기본값. sigma는 model.py 고정
# 상수와 동일, dist_eff는 init값(41.6h) -- 학습된 값과 다를 수 있으니
# 신형 npz의 저장값이 있으면 그걸 쓴다.
DEFAULT_KERNEL_SIGMA_H = 12.0
DEFAULT_DIST_EFF_H = 41.6

if len(sys.argv) > 1:
    run_dir = Path(sys.argv[1])
else:
    from config import OUTPUT_DIR as run_dir

npz_path = Path(run_dir) / "branch_decomposition" / "components.npz"
if not npz_path.exists():
    raise SystemExit(
        f"{npz_path} 없음 -- 먼저 branch_decomposition.py를 돌릴 것"
    )
out_dir = Path(run_dir) / "propagation_alignment"
out_dir.mkdir(parents=True, exist_ok=True)

npz = np.load(npz_path)
gate = npz["gate"]          # (B,20) p_t >= 0
arrival = npz["arrival"]    # (B,20) 도착 시각 [h], 음수 = 과거
speed = npz["speed"]        # (B,)   입력 wind 평균 [km/s]
n_samples = len(speed)

sigma = float(npz["kernel_sigma_hours"]) if "kernel_sigma_hours" in npz \
    else DEFAULT_KERNEL_SIGMA_H
dist_eff = float(npz["dist_eff_h"]) if "dist_eff_h" in npz \
    else DEFAULT_DIST_EFF_H
fallback = float(npz["fallback_weight"]) if "fallback_weight" in npz \
    else None

# model.py와 동일한 축: age [114..0]h, horizon [+6..+72]h.
ages = np.arange(19, -1, -1, dtype=np.float64) * STEP_HOURS   # (20,)
horizons = np.arange(1, N_HORIZON + 1, dtype=np.float64) * STEP_HOURS

# (B,20,12) forecast 격자의 kernel weight.
kernel = np.exp(
    -((horizons[None, None, :] - arrival[:, :, None]) ** 2)
    / (2.0 * sigma**2)
)
weights = gate[:, :, None] * kernel


def mass_weighted_com(w):
    """w: (n,20,12) -> horizon별 COM(hours ago). 전 샘플 질량 가중."""
    mass = w.sum(axis=(0, 1))                       # (12,)
    return (w * ages[None, :, None]).sum(axis=(0, 1)) / np.maximum(
        mass, 1e-12
    )


def theory_com(speed_kms, h):
    """ballistic: v로 h 시점에 도착하는 소스의 촬영 시점(hours ago)."""
    return dist_eff / (speed_kms / 1000.0) - h


# ---- 전체 COM + 이론 밴드 (속도 300..800 km/s -> 관측창 [0,114] clip) --
com_all = mass_weighted_com(weights)
rows = []
for j, h in enumerate(horizons):
    row = {
        "horizon_hours": h,
        "prop_com_hours_ago": com_all[j],
        "theory_low_hours_ago": np.clip(theory_com(800.0, h), 0, 114),
        "theory_high_hours_ago": np.clip(theory_com(300.0, h), 0, 114),
    }
    if fallback is not None:
        mass = weights[:, :, j].sum(axis=1)
        row["coverage_mean"] = float(
            (mass / (mass + fallback)).mean()
        )
    rows.append(row)
summary = pd.DataFrame(rows)
summary.to_csv(out_dir / "summary.csv", index=False)

# ---- 속도 3분위 (speed_ablation/branch_decomposition과 동일 정의) ----
speed_edges = np.quantile(speed, [1 / 3, 2 / 3])
group_ids = np.digitize(speed, speed_edges)
group_rows = []
for index, label in enumerate(SPEED_GROUP_LABELS):
    mask = group_ids == index
    com_group = mass_weighted_com(weights[mask])
    mean_speed = float(speed[mask].mean())
    for j, h in enumerate(horizons):
        group_rows.append({
            "group": label,
            "mean_speed_kms": mean_speed,
            "count": int(mask.sum()),
            "horizon_hours": h,
            "prop_com_hours_ago": com_group[j],
            "theory_com_hours_ago": np.clip(
                theory_com(mean_speed, h), 0, 114
            ),
        })
speed_summary = pd.DataFrame(group_rows)
speed_summary.to_csv(out_dir / "speed_summary.csv", index=False)

# ---- 판정 지표 -----------------------------------------------------
# (1) 기울기: 이론은 d COM / d h = -1 (같은 소스 집단이면 h가 6h 늦을
#     때 그 소스의 촬영 시점은 6h 얕아진다). 0에 가까우면 decoder식
#     "고정 bag 읽기"가 kernel로 재현된 것.
# (2) 레짐 분화: slow COM - fast COM. 이론값은 transit 차이
#     (~ +40h at 360 vs 510 km/s), decoder는 ~ +0.5h였다.
slope = float(np.polyfit(horizons, com_all, 1)[0])
com_slow = speed_summary[speed_summary.group == "slow"][
    "prop_com_hours_ago"
].to_numpy()
com_fast = speed_summary[speed_summary.group == "fast"][
    "prop_com_hours_ago"
].to_numpy()
slow_minus_fast = float((com_slow - com_fast).mean())
theory_gap = float(
    (
        speed_summary[speed_summary.group == "slow"][
            "theory_com_hours_ago"
        ].to_numpy()
        - speed_summary[speed_summary.group == "fast"][
            "theory_com_hours_ago"
        ].to_numpy()
    ).mean()
)

print(f"npz: {npz_path}")
print(
    f"samples: {n_samples}, sigma={sigma:.0f}h, dist_eff={dist_eff:.1f}h"
    + ("" if fallback is None else f", fallback={fallback:.2f}")
)
print("\n=== propagation kernel COM (hours ago) ===")
print(summary.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))
print("\n=== by speed tercile ===")
pivot = speed_summary.pivot(
    index="horizon_hours", columns="group",
    values="prop_com_hours_ago",
)[list(SPEED_GROUP_LABELS)]
print(pivot.to_string(float_format=lambda v: f"{v:7.2f}"))
print("\n=== 판정 ===")
print(f"  COM slope vs horizon: {slope:+.2f}  (이론 -1.0, decoder ~0.02)")
print(
    f"  slow - fast COM:      {slow_minus_fast:+.2f}h  "
    f"(이론 {theory_gap:+.1f}h, decoder ~ +0.5h)"
)

# ---- plot ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
colors = {"slow": "tab:blue", "mid": "tab:gray", "fast": "tab:red"}
for label in SPEED_GROUP_LABELS:
    part = speed_summary[speed_summary.group == label]
    ax.plot(
        part.horizon_hours, part.prop_com_hours_ago,
        marker="o", color=colors[label],
        label=f"{label} ({part.mean_speed_kms.iloc[0]:.0f} km/s)",
    )
    ax.plot(
        part.horizon_hours, part.theory_com_hours_ago,
        linestyle="--", alpha=0.5, color=colors[label],
    )
ax.fill_between(
    summary.horizon_hours,
    summary.theory_low_hours_ago,
    summary.theory_high_hours_ago,
    alpha=0.08, color="tab:green", label="theory band (300-800)",
)
ax.set_xlabel("forecast horizon (h)")
ax.set_ylabel("image COM (hours ago)")
ax.set_ylim(0, 114)
ax.set_title("propagation kernel COM vs ballistic theory (dashed)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out_dir / "propagation_com.png", dpi=150)

print(f"\nsaved to: {out_dir.resolve()}")
