"""속도 조건부 성능 분해: image gain by wind-speed group.

attention_analysis.py의 후속 실험. attention 분석에서 (1) 시점 선택이
풍속과 무관한 고정 템플릿이고 (2) fast 그룹일수록 이미지 attention
share가 낮다는 것이 확인됐다. 이 스크립트는 그것을 성능 언어로 검증한다:

               slow        mid        fast
               ────────────────────────────
  Full model    RMSE       RMSE       RMSE
  Wind-only     RMSE       RMSE       RMSE
  Persistence   RMSE       RMSE       RMSE

  Image gain = wind-only RMSE - full RMSE  (그룹별, paired bootstrap CI)

기대: slow/mid gain > 0, fast gain ≈ 0 (또는 음수)이면 "모델이 fast
wind에서 이미지를 정렬하지 못하고 실제로 활용도 못 한다"는 스토리 완성.
fast gain > 0이면 attention 슬롯 고정에도 불구하고 content 수준의
정렬이 작동한다는 반증.

설계 노트:

- Wind-only는 별도 학습 모델이 아니라 **같은 체크포인트**에서
  image_projection 출력을 0으로 만드는 inference ablation이다.
  모델이 modality_drop_prob=0.25로 학습됐기 때문에 "이미지 토큰 전부
  0 + 슬롯 유지(40토큰)"는 학습 분포 안의 조건이고, 따라서 이
  ablation은 off-distribution 페널티 없이 공정하다. use_images=False
  (토큰 자체를 제거, 20토큰)는 학습 때 본 적 없는 조건이라 쓰지 않는다.
- 속도 그룹은 attention_analysis와 동일: 입력 wind 20개 평균의 3분위.
- fast 그룹은 원래 변동성이 커서 raw RMSE 비교가 왜곡될 수 있으므로
  persistence 대비 skill(1 - RMSE/persistence)을 병기한다.
- full과 wind-only는 같은 샘플 쌍이므로 paired bootstrap으로 gain의
  신뢰구간을 계산한다.

사용법 (체크포인트/데이터가 있는 머신에서):

    python speed_ablation.py

결과물은 OUTPUT_DIR/speed_ablation/ 아래에 CSV + NPZ + PNG로 저장.
predictions.npz에 per-sample 예측을 통째로 저장하므로 로컬에서 추가
분석이 가능하다.
"""

import numpy as np
import pandas as pd
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import *
from dataset import val_loader
from inference import load_best_model

ABLATION_DIR = OUTPUT_DIR / "speed_ablation"
ABLATION_DIR.mkdir(parents=True, exist_ok=True)

N_HORIZON = 12
STEP_HOURS = 6
SPEED_GROUP_LABELS = ("slow", "mid", "fast")
SPEED_GROUP_COLORS = ("tab:blue", "tab:green", "tab:orange")
BOOTSTRAP_ITERS = 10_000


# ================================================================
# 1. Prediction passes
# ================================================================

def _zero_output_hook(module, inputs, output):
    return torch.zeros_like(output)


@torch.no_grad()
def predict_collect(model, loader, zero_images=False):
    """검증 loader 1회 pass.

    zero_images=True면 image_projection 출력을 0으로 만들어 이미지
    토큰을 zero-out한다 (슬롯/임베딩은 유지 -- 학습 시 modality
    dropout과 동일한 조건).

    returns: predictions (N,12) km/s, targets (N,12) km/s,
             speeds (N,) km/s, last_wind (N,) km/s, sample_ids
    """
    model.eval()
    handle = None
    if zero_images:
        handle = model.image_projection.register_forward_hook(
            _zero_output_hook
        )

    predictions, targets, speeds, last_wind, sample_ids = [], [], [], [], []
    try:
        for batch in loader:
            images = batch["images"].to(DEVICE, non_blocking=PIN_MEMORY)
            wind = batch["wind"].to(DEVICE, non_blocking=PIN_MEMORY)
            with torch.amp.autocast(DEVICE.type, enabled=USE_AMP):
                output = model(images, wind)
            predictions.append(output.float().cpu().numpy() * 1000.0)
            targets.append(batch["target"].numpy() * 1000.0)
            # dataset.py에서 /1000된 값이므로 km/s로 복원
            speeds.append(batch["wind"].mean(dim=1).numpy() * 1000.0)
            last_wind.append(batch["wind"][:, -1].numpy() * 1000.0)
            sample_ids.extend(batch["sample_id"])
    finally:
        if handle is not None:
            handle.remove()

    return (
        np.concatenate(predictions).astype(np.float64),
        np.concatenate(targets).astype(np.float64),
        np.concatenate(speeds).astype(np.float64),
        np.concatenate(last_wind).astype(np.float64),
        sample_ids,
    )


# ================================================================
# 2. Metrics
# ================================================================

def rmse(squared_errors):
    return float(np.sqrt(np.mean(squared_errors)))


def paired_gain_bootstrap(mse_full, mse_wind, iters=BOOTSTRAP_ITERS,
                          seed=SEED):
    """gain = RMSE(wind-only) - RMSE(full)의 paired bootstrap.

    mse_*: (n,) per-sample MSE (12개 horizon 평균). 같은 인덱스
    리샘플을 두 모델에 적용하므로 paired.

    returns: (ci_low, ci_high), p_nonpositive
    """
    rng = np.random.default_rng(seed)
    n = len(mse_full)
    indexes = rng.integers(0, n, size=(iters, n))
    gains = (
        np.sqrt(mse_wind[indexes].mean(axis=1))
        - np.sqrt(mse_full[indexes].mean(axis=1))
    )
    ci = np.percentile(gains, [2.5, 97.5])
    return (float(ci[0]), float(ci[1])), float((gains <= 0).mean())


def run_analysis(pred_full, pred_wind, pred_persistence, targets, speeds):
    """그룹별 RMSE/skill/gain 테이블과 플롯을 만든다. (원격/스모크 공용)"""
    se = {
        "full": (pred_full - targets) ** 2,
        "wind_only": (pred_wind - targets) ** 2,
        "persistence": (pred_persistence - targets) ** 2,
    }

    speed_edges = np.quantile(speeds, [1 / 3, 2 / 3])
    group_ids = np.digitize(speeds, speed_edges)
    print(
        f"speed tercile edges: {speed_edges[0]:.0f} / {speed_edges[1]:.0f} km/s"
    )

    groups = {"all": np.ones(len(speeds), dtype=bool)}
    for index, label in enumerate(SPEED_GROUP_LABELS):
        groups[label] = group_ids == index

    # ---- 그룹별 RMSE / skill / gain ----
    summary_rows, gain_rows = [], []
    for label, mask in groups.items():
        n = int(mask.sum())
        mean_speed = float(speeds[mask].mean())
        rmse_by_model = {
            name: rmse(errors[mask]) for name, errors in se.items()
        }
        for name, value in rmse_by_model.items():
            summary_rows.append({
                "group": label,
                "count": n,
                "mean_speed_kms": mean_speed,
                "model": name,
                "rmse_km_s": value,
                "skill_vs_persistence":
                    1.0 - value / rmse_by_model["persistence"],
            })

        mse_full = se["full"][mask].mean(axis=1)
        mse_wind = se["wind_only"][mask].mean(axis=1)
        gain = rmse_by_model["wind_only"] - rmse_by_model["full"]
        (ci_low, ci_high), p_nonpositive = paired_gain_bootstrap(
            mse_full, mse_wind
        )
        gain_rows.append({
            "group": label,
            "count": n,
            "mean_speed_kms": mean_speed,
            "image_gain_km_s": gain,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_nonpositive": p_nonpositive,
            "gain_pct_of_persistence":
                100.0 * gain / rmse_by_model["persistence"],
        })

    summary = pd.DataFrame(summary_rows)
    gains = pd.DataFrame(gain_rows)
    summary.to_csv(ABLATION_DIR / "speed_ablation.csv", index=False)
    gains.to_csv(ABLATION_DIR / "image_gain.csv", index=False)

    # ---- horizon x 그룹 분해 ----
    horizon_rows = []
    for label, mask in groups.items():
        for j in range(N_HORIZON):
            row = {
                "group": label,
                "horizon_hours": (j + 1) * STEP_HOURS,
            }
            for name, errors in se.items():
                row[f"rmse_{name}"] = rmse(errors[mask][:, j])
            row["image_gain_km_s"] = (
                row["rmse_wind_only"] - row["rmse_full"]
            )
            horizon_rows.append(row)
    by_horizon = pd.DataFrame(horizon_rows)
    by_horizon.to_csv(ABLATION_DIR / "gain_by_horizon.csv", index=False)

    # ---- 사용자 스케치 형태의 테이블 출력 ----
    order = [l for l in (*SPEED_GROUP_LABELS, "all")]
    rmse_pivot = summary.pivot(
        index="model", columns="group", values="rmse_km_s"
    )[order].reindex(["full", "wind_only", "persistence"])
    skill_pivot = summary.pivot(
        index="model", columns="group", values="skill_vs_persistence"
    )[order].reindex(["full", "wind_only"])

    print("\n=== RMSE (km/s) ===")
    print(rmse_pivot.to_string(float_format=lambda v: f"{v:8.2f}"))
    print("\n=== skill vs persistence (1 - RMSE/persistence) ===")
    print(skill_pivot.to_string(float_format=lambda v: f"{v:8.3f}"))
    print("\n=== image gain (wind-only RMSE - full RMSE) ===")
    for _, row in gains.set_index("group").loc[order].iterrows():
        print(
            f"  {row.name:5s}: {row.image_gain_km_s:+6.2f} km/s "
            f"[{row.ci_low:+6.2f}, {row.ci_high:+6.2f}] "
            f"({row.gain_pct_of_persistence:+.1f}% of persistence, "
            f"P(gain<=0)={row.p_nonpositive:.3f}, n={int(row['count'])})"
        )

    # ---- plots ----
    plot_gain_bars(gains)
    plot_gain_by_horizon(by_horizon)

    return summary, gains, by_horizon


# ================================================================
# 3. Plots
# ================================================================

def plot_gain_bars(gains):
    frame = gains.set_index("group").loc[list(SPEED_GROUP_LABELS)]
    x = np.arange(len(frame))
    values = frame.image_gain_km_s.to_numpy()
    err = np.vstack([
        values - frame.ci_low.to_numpy(),
        frame.ci_high.to_numpy() - values,
    ])

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.bar(x, values, yerr=err, capsize=5, color=SPEED_GROUP_COLORS,
           alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"{label}\n(v={frame.mean_speed_kms[label]:.0f} km/s)"
        for label in frame.index
    ], fontsize=9)
    ax.set_ylabel("image gain: wind-only RMSE - full RMSE (km/s)")
    ax.set_title("does the image pathway help, by wind speed?\n"
                 "(paired bootstrap 95% CI)")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(ABLATION_DIR / "image_gain_by_group.png", dpi=150)
    plt.close(fig)


def plot_gain_by_horizon(by_horizon):
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for label, color in zip(SPEED_GROUP_LABELS, SPEED_GROUP_COLORS):
        frame = by_horizon[by_horizon.group == label]
        ax.plot(
            frame.horizon_hours, frame.image_gain_km_s,
            "o-", color=color, label=label,
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("forecast horizon (hours)")
    ax.set_ylabel("image gain (km/s)")
    ax.set_title("image gain by horizon and wind speed")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(ABLATION_DIR / "image_gain_by_horizon.png", dpi=150)
    plt.close(fig)


# ================================================================
# 4. Main
# ================================================================

if __name__ == "__main__":
    model = load_best_model()
    if not model.use_images:
        raise SystemExit("use_images=False 모델은 ablation 대상이 아님")

    print("\n[full model pass]")
    pred_full, targets, speeds, last_wind, sample_ids = predict_collect(
        model, val_loader, zero_images=False
    )
    print("[wind-only pass (image tokens zeroed)]")
    pred_wind, targets_check, _, _, ids_check = predict_collect(
        model, val_loader, zero_images=True
    )
    # 두 pass가 같은 순서/샘플이라는 sanity check
    assert np.array_equal(targets, targets_check)
    assert list(sample_ids) == list(ids_check)

    pred_persistence = np.repeat(last_wind[:, None], N_HORIZON, axis=1)

    np.savez(
        ABLATION_DIR / "predictions.npz",
        pred_full=pred_full,
        pred_wind_only=pred_wind,
        pred_persistence=pred_persistence,
        targets=targets,
        speed_kms=speeds,
        sample_ids=np.asarray(sample_ids),
    )

    run_analysis(pred_full, pred_wind, pred_persistence, targets, speeds)
    print(f"\nsaved to: {ABLATION_DIR.resolve()}")
