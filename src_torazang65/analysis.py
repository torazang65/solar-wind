"""v7 통합 분석: 체크포인트 하나에 대한 전 진단을 한 번에.

구 스크립트 5개를 대체한다 (git 히스토리에 보존, v6 이하 런은 해당
커밋의 스크립트로 분석할 것 -- v7 체크포인트는 소스 축이 (B,20,2,4)라
구 스크립트와 호환되지 않고, 반대도 마찬가지):

  diagnose.py              -> [1] overview, [6] split 비교, --with-train
  speed_ablation.py        -> [2] image ablation (v7: force_image_drop
                              스위치 사용 -- projection hook은 readout
                              경로를 못 껐다)
  branch_decomposition.py  -> [3] decomposition, [5] mechanism
  propagation_alignment.py -> [4] alignment. v7 재정의: 자전 기하에서는
                              소스가 창 전체에 걸쳐 보이므로(궤적)
                              촬영시점(sighting) COM은 판별력을 잃는다.
                              대신 소스별 함의 출발시점
                                  launch_age = age + lon/omega
                              의 COM을 쓴다 -- 참 소스라면 어느 프레임
                              에서 보였든 launch_age = tau*(y) - u로
                              동일하므로 이론 곡선(기울기 -1, 속도군
                              분화 = tau* 차이)이 그대로 성립한다.
  attention_analysis.py    -> 폐기 (decoder가 v6a에서 제거됨)

사용법 (체크포인트/데이터가 있는 머신에서):

    python analysis.py                # val 분석 전체
    python analysis.py --with-train   # + clean-train 간격 (train 1 pass)

모델 full 1 pass + image-drop 1 pass로 전 섹션을 계산한다 (구 스크립트
들이 각자 pass를 돌리던 중복 제거). 산출물: OUTPUT_DIR/analysis/ 아래
CSV/PNG/NPZ + 콘솔 리포트. 마지막 "판정" 블록이 config.py v7 주석의
판정 기준 (1)~(5)에 대응하는 숫자를 모아 출력한다.
"""

import argparse

import numpy as np
import pandas as pd
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import *
from dataset import (
    train_inputs, train_loader, train_targets, val_inputs, val_loader,
)
from inference import load_best_model

ANALYSIS_DIR = OUTPUT_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

N_HORIZON = 12
STEP_HOURS = 6.0
SPEED_GROUP_LABELS = ("slow", "mid", "fast")
SPEED_GROUP_COLORS = ("tab:blue", "tab:green", "tab:orange")
SURGE_THRESHOLD_KMS = 100.0
BOOTSTRAP_ITERS = 10_000
WORST_K = 15


# ================================================================
# 1. Collection (full 1 pass + image-drop 1 pass)
# ================================================================

@torch.no_grad()
def collect(model, loader, image_drop=False, stash=True):
    """loader 1 pass, fp32 (성분 일치 검증 목적이라 autocast 없음).

    image_drop=True면 model.force_image_drop으로 이미지 스트림을 통째
    로 0으로 -- 학습의 modality drop과 동일 조건이라 공정한 ablation.
    stash=False면 예측만 모은다 (drop pass용).
    """
    model.eval()
    model.force_image_drop = image_drop
    out = {k: [] for k in (
        "pred", "target", "wind", "speed", "last_wind", "sample_id",
        "base", "v_img", "alpha", "surge_prob",
        "src_speed", "arrival", "gate", "lon", "coverage",
    )}
    try:
        for batch in loader:
            images = batch["images"].to(DEVICE, non_blocking=PIN_MEMORY)
            wind = batch["wind"].to(DEVICE, non_blocking=PIN_MEMORY)
            pred = model(images, wind)
            out["pred"].append(pred.float().cpu().numpy())
            out["target"].append(batch["target"].numpy())
            if not stash:
                continue
            out["wind"].append(batch["wind"].numpy())
            out["speed"].append(batch["wind"].mean(dim=1).numpy())
            out["last_wind"].append(batch["wind"][:, -1].numpy())
            out["sample_id"].append(np.asarray(batch["sample_id"]))
            out["base"].append(model.last_base.float().cpu().numpy())
            out["v_img"].append(
                model.last_v_image_future.float().cpu().numpy()
            )
            out["alpha"].append(
                model.last_fusion_alpha.float().cpu().numpy()
            )
            out["surge_prob"].append(
                model.last_surge_prob.float().cpu().numpy()
            )
            out["src_speed"].append(
                model.last_source_speed_kms.float().cpu().numpy()
            )
            out["arrival"].append(
                model.last_arrival_hours.float().cpu().numpy()
            )
            out["gate"].append(
                model.last_source_gate.float().cpu().numpy()
            )
            out["lon"].append(
                model.last_source_lon_deg.float().cpu().numpy()
            )
            out["coverage"].append(
                model.last_coverage.float().cpu().numpy()
            )
    finally:
        model.force_image_drop = False

    a = {
        k: np.concatenate(v)
        for k, v in out.items() if v
    }
    # /1000 스케일 -> km/s (src_speed는 모델이 이미 km/s로 stash).
    for k in ("pred", "target", "wind", "speed", "last_wind",
              "base", "v_img"):
        if k in a:
            a[k] = a[k].astype(np.float64) * 1000.0
    return a


# ================================================================
# 2. Helpers
# ================================================================

def rmse(squared_errors):
    return float(np.sqrt(np.mean(squared_errors)))


def paired_gain_bootstrap(mse_baseline, mse_variant,
                          iters=BOOTSTRAP_ITERS, seed=SEED):
    """gain = RMSE(baseline) - RMSE(variant)의 paired bootstrap CI."""
    rng = np.random.default_rng(seed)
    n = len(mse_baseline)
    indexes = rng.integers(0, n, size=(iters, n))
    gains = (
        np.sqrt(mse_baseline[indexes].mean(axis=1))
        - np.sqrt(mse_variant[indexes].mean(axis=1))
    )
    ci = np.percentile(gains, [2.5, 97.5])
    return (float(ci[0]), float(ci[1])), float((gains <= 0).mean())


def weighted_corr(x, y, w):
    w = w / w.sum()
    mx, my = (w * x).sum(), (w * y).sum()
    cov = (w * (x - mx) * (y - my)).sum()
    sx = np.sqrt((w * (x - mx) ** 2).sum())
    sy = np.sqrt((w * (y - my) ** 2).sum())
    return float(cov / (sx * sy + 1e-12))


def banner(title):
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


def speed_groups(speed):
    """입력 wind 평균 3분위 (전 구 스크립트와 동일 정의)."""
    edges = np.quantile(speed, [1 / 3, 2 / 3])
    ids = np.digitize(speed, edges)
    groups = {"all": np.ones(len(speed), dtype=bool)}
    for index, label in enumerate(SPEED_GROUP_LABELS):
        groups[label] = ids == index
    return groups


# ================================================================
# [1] Overview: 베이스라인 / horizon / event share / worst-k
# ================================================================

def section_overview(a, pred_persistence):
    banner("[1] overview: 앵커 베이스라인 (val 전체 RMSE, km/s)")
    se_model = (a["pred"] - a["target"]) ** 2
    se_persistence = (pred_persistence - a["target"]) ** 2
    climatology = train_targets.mean(axis=0)  # raw km/s
    se_climatology = (climatology[None] - a["target"]) ** 2
    print(f"model                    : {rmse(se_model):8.3f}")
    print(f"persistence (wind[-1])   : {rmse(se_persistence):8.3f}")
    print(f"climatology (train mean) : {rmse(se_climatology):8.3f}")

    delta_true = a["target"] - a["last_wind"][:, None]
    delta_pred = a["pred"] - a["last_wind"][:, None]
    rows = []
    for j in range(N_HORIZON):
        rows.append({
            "horizon_hours": (j + 1) * STEP_HOURS,
            "rmse_model": rmse(se_model[:, j]),
            "rmse_persistence": rmse(se_persistence[:, j]),
            "rmse_climatology": rmse(se_climatology[:, j]),
            # 이벤트 축소비율: 1보다 많이 작으면 밋밋한 예측으로 후퇴.
            "shrinkage": float(
                np.std(delta_pred[:, j]) / np.std(delta_true[:, j])
            ),
        })
    overview = pd.DataFrame(rows)
    overview.to_csv(ANALYSIS_DIR / "overview_by_horizon.csv",
                    index=False)
    print("\n" + overview.to_string(
        index=False, float_format=lambda v: f"{v:8.3f}"
    ))

    # 이벤트 기여 + worst-k
    delta_max = np.abs(delta_true).max(axis=1)
    is_event = delta_max >= SURGE_THRESHOLD_KMS
    event_share = se_model[is_event].sum() / se_model.sum()
    print(
        f"\nevent(n={int(is_event.sum())}) RMSE "
        f"{rmse(se_model[is_event]):.2f} vs persistence "
        f"{rmse(se_persistence[is_event]):.2f} / "
        f"quiet(n={int((~is_event).sum())}) "
        f"{rmse(se_model[~is_event]):.2f} vs "
        f"{rmse(se_persistence[~is_event]):.2f}"
    )
    print(f"전체 제곱오차 중 이벤트 기여: {event_share:.1%}")

    per_sample = pd.DataFrame({
        "sample_id": a["sample_id"],
        "rmse_model": np.sqrt(se_model.mean(axis=1)),
        "rmse_persistence": np.sqrt(se_persistence.mean(axis=1)),
        "last_wind": a["last_wind"],
        "delta_max": delta_max,
        "is_event": is_event,
    }).sort_values("rmse_model", ascending=False)
    per_sample.to_csv(ANALYSIS_DIR / "per_sample.csv", index=False)
    print(f"\nworst-{WORST_K}:")
    print(per_sample.head(WORST_K).to_string(
        index=False, float_format=lambda v: f"{v:.1f}"
    ))


# ================================================================
# [2] Image ablation (full vs image-drop vs persistence)
# ================================================================

def section_ablation(a, pred_drop, pred_persistence, groups):
    banner("[2] image ablation: gain = image-drop RMSE - full RMSE")
    se = {
        "full": (a["pred"] - a["target"]) ** 2,
        "image_drop": (pred_drop - a["target"]) ** 2,
        "persistence": (pred_persistence - a["target"]) ** 2,
    }
    surge = a["target"].max(axis=1) - a["last_wind"]
    conditions = {
        "any": np.ones(len(surge), dtype=bool),
        "surge": surge > SURGE_THRESHOLD_KMS,
        "quiet": surge <= SURGE_THRESHOLD_KMS,
    }

    rows = []
    for group_label, group_mask in groups.items():
        for condition_label, condition_mask in conditions.items():
            mask = group_mask & condition_mask
            n = int(mask.sum())
            if n < 30:
                continue
            rmse_by = {
                name: rmse(errors[mask]) for name, errors in se.items()
            }
            ci, p = paired_gain_bootstrap(
                se["image_drop"][mask].mean(axis=1),
                se["full"][mask].mean(axis=1),
            )
            rows.append({
                "group": group_label,
                "condition": condition_label,
                "count": n,
                "mean_speed_kms": float(a["speed"][mask].mean()),
                **{f"rmse_{k}": v for k, v in rmse_by.items()},
                "skill_vs_persistence":
                    1.0 - rmse_by["full"] / rmse_by["persistence"],
                "image_gain_km_s":
                    rmse_by["image_drop"] - rmse_by["full"],
                "ci_low": ci[0],
                "ci_high": ci[1],
                "p_nonpositive": p,
            })
    gains = pd.DataFrame(rows)
    gains.to_csv(ANALYSIS_DIR / "image_gain.csv", index=False)
    for _, row in gains.iterrows():
        print(
            f"  {row.group:5s} {row.condition:6s} (n={int(row['count']):4d}): "
            f"full {row.rmse_full:7.2f}  drop {row.rmse_image_drop:7.2f}  "
            f"gain {row.image_gain_km_s:+7.2f} "
            f"[{row.ci_low:+6.2f}, {row.ci_high:+6.2f}] "
            f"P(<=0)={row.p_nonpositive:.3f}"
        )

    # horizon x 그룹 gain + plots
    horizon_rows = []
    for group_label, group_mask in groups.items():
        for j in range(N_HORIZON):
            horizon_rows.append({
                "group": group_label,
                "horizon_hours": (j + 1) * STEP_HOURS,
                "image_gain_km_s": (
                    rmse(se["image_drop"][group_mask][:, j])
                    - rmse(se["full"][group_mask][:, j])
                ),
            })
    by_horizon = pd.DataFrame(horizon_rows)
    by_horizon.to_csv(ANALYSIS_DIR / "image_gain_by_horizon.csv",
                      index=False)

    part = gains[gains.condition == "any"].set_index("group")
    part = part.loc[[g for g in SPEED_GROUP_LABELS if g in part.index]]
    values = part.image_gain_km_s.to_numpy()
    err = np.vstack([
        values - part.ci_low.to_numpy(),
        part.ci_high.to_numpy() - values,
    ])
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.bar(np.arange(len(part)), values, yerr=err, capsize=5,
           color=SPEED_GROUP_COLORS[: len(part)], alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(part)))
    ax.set_xticklabels([
        f"{label}\n(v={part.mean_speed_kms[label]:.0f})"
        for label in part.index
    ], fontsize=9)
    ax.set_ylabel("image gain (km/s)")
    ax.set_title("image gain by wind speed (95% CI)")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(ANALYSIS_DIR / "image_gain_by_group.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for label, color in zip(SPEED_GROUP_LABELS, SPEED_GROUP_COLORS):
        frame = by_horizon[by_horizon.group == label]
        ax.plot(frame.horizon_hours, frame.image_gain_km_s,
                "o-", color=color, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("forecast horizon (hours)")
    ax.set_ylabel("image gain (km/s)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(ANALYSIS_DIR / "image_gain_by_horizon.png", dpi=150)
    plt.close(fig)
    return gains


# ================================================================
# [3] Decomposition: base / base+shrink / full(=base+prop)
# ================================================================

def section_decomposition(a, model, groups):
    banner("[3] decomposition: pred = base + alpha*(v_img - base)")
    climatology_kms = float(model.climatology.detach()) * 1000.0
    predictions = {
        "base": a["base"],
        # shrink 반사실: v_img를 상수 climatology로 바꿔 "정렬 없이
        # 상수로 당기기"만 남긴 것. prop - shrink = 정렬의 순수 기여.
        "base+shrink":
            a["base"] + a["alpha"] * (climatology_kms - a["base"]),
        "full": a["base"] + a["alpha"] * (a["v_img"] - a["base"]),
    }
    reassembly_error = np.abs(predictions["full"] - a["pred"]).max()
    assert reassembly_error < 1.0, (
        f"재조립 불일치 {reassembly_error:.3f} km/s -- "
        "model.py stash와 조립식이 어긋남"
    )
    se = {
        name: (pred - a["target"]) ** 2
        for name, pred in predictions.items()
    }
    surge = a["target"].max(axis=1) - a["last_wind"]
    conditions = {
        "any": np.ones(len(surge), dtype=bool),
        "surge": surge > SURGE_THRESHOLD_KMS,
        "quiet": surge <= SURGE_THRESHOLD_KMS,
    }

    rows = []
    for group_label, group_mask in groups.items():
        for condition_label, condition_mask in conditions.items():
            mask = group_mask & condition_mask
            n = int(mask.sum())
            if n < 30:
                continue
            rmse_by = {
                name: rmse(errors[mask]) for name, errors in se.items()
            }
            prop_ci, prop_p = paired_gain_bootstrap(
                se["base"][mask].mean(axis=1),
                se["full"][mask].mean(axis=1),
            )
            align_ci, align_p = paired_gain_bootstrap(
                se["base+shrink"][mask].mean(axis=1),
                se["full"][mask].mean(axis=1),
            )
            rows.append({
                "group": group_label,
                "condition": condition_label,
                "count": n,
                **{f"rmse_{k}": v for k, v in rmse_by.items()},
                "prop_own_gain": rmse_by["base"] - rmse_by["full"],
                "prop_ci_low": prop_ci[0],
                "prop_ci_high": prop_ci[1],
                "prop_p_nonpositive": prop_p,
                "alignment_gain":
                    rmse_by["base+shrink"] - rmse_by["full"],
                "align_ci_low": align_ci[0],
                "align_ci_high": align_ci[1],
                "align_p_nonpositive": align_p,
                "alpha_mean": float(a["alpha"][mask].mean()),
                "v_img_std_kms": float(a["v_img"][mask].std()),
                "surge_prob_mean": float(a["surge_prob"][mask].mean()),
            })
    decomposition = pd.DataFrame(rows)
    decomposition.to_csv(ANALYSIS_DIR / "component_gain.csv",
                         index=False)
    for _, row in decomposition.iterrows():
        print(
            f"  {row.group:5s} {row.condition:6s} (n={int(row['count']):4d}): "
            f"prop {row.prop_own_gain:+7.2f} "
            f"[{row.prop_ci_low:+6.2f}, {row.prop_ci_high:+6.2f}]  "
            f"align {row.alignment_gain:+7.2f} "
            f"[{row.align_ci_low:+6.2f}, {row.align_ci_high:+6.2f}] "
            f"P(<=0)={row.align_p_nonpositive:.3f}  "
            f"alpha {row.alpha_mean:.2f}  "
            f"p_surge {row.surge_prob_mean:.2f}"
        )
    return decomposition


# ================================================================
# [4] Alignment (v7): launch-age COM / 궤적 추적 / align KL
# ================================================================

def section_alignment(a, model, groups):
    banner("[4] alignment: launch-age COM vs ballistic 이론")
    sigma = float(model.kernel_sigma_hours)
    dist_eff = float(
        30.0 + 25.0 * torch.sigmoid(model.dist_eff_raw.detach())
    )
    fallback = float(torch.nn.functional.softplus(
        model.fallback_weight_raw.detach()
    ))
    omega = float(model.omega_deg_per_hour)
    cell_lon = model.cell_lon_deg.cpu().numpy()

    n = len(a["speed"])
    n_src = 20 * 2 * 4
    ages = np.repeat(
        np.arange(19, -1, -1, dtype=np.float64) * STEP_HOURS, 8
    )  # (160,) 소스별 촬영 age
    horizons = np.arange(1, N_HORIZON + 1, dtype=np.float64) * STEP_HOURS

    arrival = a["arrival"].reshape(n, n_src).astype(np.float64)
    gate = a["gate"].reshape(n, n_src).astype(np.float64)
    lon = a["lon"].reshape(n, n_src).astype(np.float64)
    src_speed = a["src_speed"].reshape(n, n_src).astype(np.float64)

    # forecast 격자 kernel weight (B,160,12)
    kernel = np.exp(
        -((horizons[None, None, :] - arrival[:, :, None]) ** 2)
        / (2.0 * sigma ** 2)
    )
    weights = gate[:, :, None] * kernel

    # 소스별 함의 출발시점 (hours ago). 참 소스는 어느 프레임에서
    # 보였든 launch_age = tau*(y) - u로 동일 -- 모듈 docstring 참고.
    launch_age = ages[None, :] + lon / omega  # (B,160)

    def launch_com(w, age):
        mass = w.sum(axis=(0, 1))
        return (w * age[:, :, None]).sum(axis=(0, 1)) / np.maximum(
            mass, 1e-12
        )

    rows, com_by_group = [], {}
    for label in ("all",) + SPEED_GROUP_LABELS:
        mask = groups[label]
        com = launch_com(weights[mask], launch_age[mask])
        com_by_group[label] = com
        mean_speed = float(a["speed"][mask].mean())
        theory = dist_eff / (mean_speed / 1000.0) - horizons
        for j, h in enumerate(horizons):
            rows.append({
                "group": label,
                "mean_speed_kms": mean_speed,
                "count": int(mask.sum()),
                "horizon_hours": h,
                "launch_com_hours_ago": com[j],
                "theory_hours_ago": theory[j],
            })
    alignment = pd.DataFrame(rows)
    alignment.to_csv(ANALYSIS_DIR / "alignment_by_speed.csv",
                     index=False)

    pivot = alignment.pivot(
        index="horizon_hours", columns="group",
        values="launch_com_hours_ago",
    )[list(SPEED_GROUP_LABELS)]
    print(pivot.to_string(float_format=lambda v: f"{v:7.2f}"))

    slope = float(np.polyfit(horizons, com_by_group["all"], 1)[0])
    slow_minus_fast = float(
        (com_by_group["slow"] - com_by_group["fast"]).mean()
    )
    theory_gap = float((
        alignment[alignment.group == "slow"].theory_hours_ago.to_numpy()
        - alignment[alignment.group == "fast"].theory_hours_ago.to_numpy()
    ).mean())

    # 궤적 추적: (sample, u) 안에서 lon vs age의 가중 기울기.
    # 같은 소스의 sighting들은 lon = -omega*(u - tau* + age)를 따르므로
    # 이론값은 -omega (= -0.55 deg/h). 0이면 "궤적이 아니라 한 덩어리".
    frame_ages = ages[None, :, None]
    w_sum = weights.sum(axis=1, keepdims=True)
    valid_u = w_sum[:, 0, :] > 1e-6
    mean_age = (weights * frame_ages).sum(axis=1, keepdims=True) / \
        np.maximum(w_sum, 1e-12)
    mean_lon = (weights * lon[:, :, None]).sum(axis=1, keepdims=True) / \
        np.maximum(w_sum, 1e-12)
    cov = (weights * (frame_ages - mean_age)
           * (lon[:, :, None] - mean_lon)).sum(axis=1)
    var = (weights * (frame_ages - mean_age) ** 2).sum(axis=1)
    slopes = cov / np.maximum(var, 1e-9)
    rotation_slope = float(slopes[valid_u & (var > 36.0)].mean())

    # 같은 프레임 안 cell 간 arrival 분화 (gate 가중 std). v6까지는
    # 구조상 0이었다 -- 커져야 경도 분해가 실제로 일한다는 뜻.
    arrival_cells = a["arrival"].reshape(n, 20, 8).astype(np.float64)
    gate_cells = a["gate"].reshape(n, 20, 8).astype(np.float64)
    g_sum = gate_cells.sum(axis=2)
    m = (gate_cells * arrival_cells).sum(axis=2) / \
        np.maximum(g_sum, 1e-12)
    var_cells = (gate_cells * (arrival_cells - m[:, :, None]) ** 2
                 ).sum(axis=2) / np.maximum(g_sum, 1e-12)
    within_frame_spread = float(np.sqrt(var_cells).mean())

    # align KL (train.py alignment_kl의 numpy 재현, 전체 25 격자).
    time_grid = np.concatenate([
        np.arange(-12, 1, dtype=np.float64) * STEP_HOURS, horizons,
    ])
    kernel25 = np.exp(
        -((time_grid[None, None, :] - arrival[:, :, None]) ** 2)
        / (2.0 * sigma ** 2)
    )
    w25 = gate[:, :, None] * kernel25 + 1e-8
    w25 = w25 / w25.sum(axis=1, keepdims=True)
    y_series = np.concatenate(
        [a["wind"][:, 7:], a["target"]], axis=1
    ) / 1000.0
    tau_star = dist_eff / np.clip(y_series, 0.2, 1.2)  # (B,25)
    frame_age = np.arange(19, -1, -1, dtype=np.float64) * STEP_HOURS
    delta_t = (
        time_grid[None, None, :] - tau_star[:, None, :]
        + frame_age[None, :, None]
    )  # (B,20,25)
    lon_star = -omega * delta_t
    gaussian = np.exp(
        -((cell_lon[None, None, :, None]
           - lon_star[:, :, None, :]) ** 2)
        / (2.0 * ALIGN_SIGMA_DEG ** 2)
    ) * (np.abs(lon_star)[:, :, None, :] <= 90.0)  # (B,20,4,25)
    q = np.repeat(gaussian[:, :, None, :, :], 2, axis=2) / 2.0
    q = q.reshape(n, n_src, 25)
    q_mass = q.sum(axis=1)
    valid = q_mass > 1e-6
    q = q / np.maximum(q_mass, 1e-6)[:, None, :]
    kl = (q * (np.log(np.maximum(q, 1e-12)) - np.log(w25))).sum(axis=1)
    align_kl = float(kl[valid].mean())

    metrics = pd.DataFrame([{
        "dist_eff_h": dist_eff,
        "fallback_weight": fallback,
        "com_slope_per_hour": slope,
        "slow_minus_fast_com_h": slow_minus_fast,
        "theory_gap_h": theory_gap,
        "rotation_slope_deg_per_h": rotation_slope,
        "within_frame_arrival_spread_h": within_frame_spread,
        "align_kl": align_kl,
        "align_valid_share": float(valid.mean()),
        "gate_mean": float(gate.mean()),
        "coverage_future_mean": float(a["coverage"][:, 13:].mean()),
        "src_speed_std_kms": float(src_speed.std()),
    }])
    metrics.to_csv(ANALYSIS_DIR / "alignment_metrics.csv", index=False)

    print(f"\n  COM slope vs horizon        : {slope:+.2f}  (이론 -1.0)")
    print(
        f"  slow - fast launch COM      : {slow_minus_fast:+.2f}h  "
        f"(이론 {theory_gap:+.1f}h, v6b sighting-COM은 ~7h)"
    )
    print(
        f"  rotation tracking slope     : {rotation_slope:+.3f} deg/h  "
        f"(이론 {-omega:+.3f})"
    )
    print(
        f"  within-frame arrival spread : {within_frame_spread:.1f}h  "
        f"(v6까지 구조상 0)"
    )
    print(
        f"  align KL (val)              : {align_kl:.3f}  "
        f"(valid {float(valid.mean()):.2f}, uniform ~{np.log(160):.1f})"
    )

    # plot: launch-age COM vs 이론
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"slow": "tab:blue", "mid": "tab:gray", "fast": "tab:red"}
    for label in SPEED_GROUP_LABELS:
        part = alignment[alignment.group == label]
        ax.plot(part.horizon_hours, part.launch_com_hours_ago,
                marker="o", color=colors[label],
                label=f"{label} ({part.mean_speed_kms.iloc[0]:.0f} km/s)")
        ax.plot(part.horizon_hours, part.theory_hours_ago,
                linestyle="--", alpha=0.5, color=colors[label])
    ax.set_xlabel("forecast horizon (h)")
    ax.set_ylabel("implied launch age (hours ago)")
    ax.set_title("launch-age COM vs ballistic theory (dashed)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "launch_com.png", dpi=150)
    plt.close(fig)
    return metrics


# ================================================================
# [5] Mechanism: corr(s_t, wind@arrival)
# ================================================================

def section_mechanism(a, groups):
    banner("[5] mechanism: corr(source speed, 도착 시각의 실측 wind)")
    n = len(a["speed"])
    n_src = 20 * 2 * 4
    arrival = a["arrival"].reshape(n, n_src).astype(np.float64)
    gate = a["gate"].reshape(n, n_src).astype(np.float64)
    src_speed = a["src_speed"].reshape(n, n_src).astype(np.float64)

    time_grid = np.concatenate([
        np.arange(-19, 1) * STEP_HOURS,
        np.arange(1, 13) * STEP_HOURS,
    ])
    series = np.concatenate([a["wind"], a["target"]], axis=1)
    observed = np.stack([
        np.interp(arrival[i], time_grid, series[i]) for i in range(n)
    ])
    in_window = (
        (arrival >= time_grid[0]) & (arrival <= time_grid[-1])
    )

    rows = []
    for scope_label, scope in (
        ("all_sources", in_window),
        ("future_arrivals", in_window & (arrival > 0)),
        ("past_arrivals", in_window & (arrival <= 0)),
    ):
        w = (gate * scope).ravel()
        if w.sum() <= 0:
            continue
        rows.append({
            "scope": scope_label,
            "corr": weighted_corr(
                src_speed.ravel(), observed.ravel(), w
            ),
        })
    # within-sample: 샘플별 가중 평균을 빼 레짐 수준 효과 제거. 이
    # 값이 양수여야 "샘플 안에서 어느 소스가 어느 시점 wind를
    # 만드는지"를 읽는다는 증거다.
    w = gate * in_window
    w_sum = w.sum(axis=1, keepdims=True)
    has = w_sum[:, 0] > 0
    x = src_speed[has] - (w[has] * src_speed[has]).sum(
        axis=1, keepdims=True) / w_sum[has]
    y = observed[has] - (w[has] * observed[has]).sum(
        axis=1, keepdims=True) / w_sum[has]
    rows.append({
        "scope": "within_sample",
        "corr": weighted_corr(x.ravel(), y.ravel(), w[has].ravel()),
    })
    mechanism = pd.DataFrame(rows)
    mechanism.to_csv(ANALYSIS_DIR / "mechanism.csv", index=False)
    print(mechanism.to_string(index=False,
                              float_format=lambda v: f"{v:7.3f}"))
    return mechanism


# ================================================================
# [6] split 비교 + (--with-train) clean-train 간격
# ================================================================

def section_split(a, model, with_train):
    banner("[6] train/val 구성 비교")
    print(f"샘플 수: train={len(train_inputs)}  val={len(val_inputs)}")
    print(f"{'':>18} {'train':>10} {'val':>10}")
    for label, train_value, val_value in (
        ("target mean", train_targets.mean(), a["target"].mean()),
        ("target std", train_targets.std(), a["target"].std()),
        ("target p99", np.percentile(train_targets, 99),
         np.percentile(a["target"], 99)),
        ("last_wind mean", None, a["last_wind"].mean()),
    ):
        train_text = "" if train_value is None else f"{train_value:10.1f}"
        print(f"{label:>18} {train_text:>10} {val_value:>10.1f}")

    if not with_train:
        print("\n(clean-train 간격은 --with-train으로 실행 시 계산)")
        return
    print("\ncollecting clean-train predictions ...", flush=True)
    t = collect(model, train_loader, stash=False)
    train_rmse = rmse((t["pred"] - t["target"]) ** 2)
    val_rmse = rmse((a["pred"] - a["target"]) ** 2)
    print(f"train RMSE (clean) : {train_rmse:8.3f}")
    print(f"val   RMSE         : {val_rmse:8.3f}")
    print(f"진짜 train-val 간격 : {val_rmse - train_rmse:8.3f}")


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-train", action="store_true",
                        help="clean-train 간격도 계산 (train 1 pass 추가)")
    args = parser.parse_args()

    model = load_best_model()
    if not model.use_images:
        raise SystemExit("use_images=False 모델은 분석 대상이 아님")

    print("[full pass]", flush=True)
    a = collect(model, val_loader)
    print("[image-drop pass]", flush=True)
    d = collect(model, val_loader, image_drop=True, stash=False)
    # collect는 stash 여부와 무관하게 pred/target을 km/s로 스케일한다.
    assert np.array_equal(a["target"], d["target"])
    pred_drop = d["pred"]
    pred_persistence = np.repeat(
        a["last_wind"][:, None], N_HORIZON, axis=1
    )
    groups = speed_groups(a["speed"])

    section_overview(a, pred_persistence)
    gains = section_ablation(a, pred_drop, pred_persistence, groups)
    decomposition = section_decomposition(a, model, groups)
    metrics = section_alignment(a, model, groups)
    mechanism = section_mechanism(a, groups)
    section_split(a, model, args.with_train)

    # 로컬 재분석용 dump (원격 학습 / 로컬 분석 워크플로).
    np.savez(
        ANALYSIS_DIR / "dump.npz",
        pred_full=a["pred"], pred_image_drop=pred_drop,
        pred_persistence=pred_persistence,
        target=a["target"], wind=a["wind"], speed=a["speed"],
        last_wind=a["last_wind"], sample_ids=a["sample_id"],
        base=a["base"], v_img=a["v_img"], alpha=a["alpha"],
        surge_prob=a["surge_prob"], coverage=a["coverage"],
        src_speed=a["src_speed"], arrival=a["arrival"],
        gate=a["gate"], lon=a["lon"],
        kernel_sigma_hours=float(model.kernel_sigma_hours),
        dist_eff_h=float(
            30.0 + 25.0 * torch.sigmoid(model.dist_eff_raw.detach())
        ),
        fallback_weight=float(torch.nn.functional.softplus(
            model.fallback_weight_raw.detach()
        )),
        climatology_kms=float(model.climatology.detach()) * 1000.0,
        omega_deg_per_hour=float(model.omega_deg_per_hour),
    )

    # ---- 판정 블록 (config.py v7 기준 (1)~(5) 대응 숫자) ----
    banner("판정 (config.py v7 기준)")
    m = metrics.iloc[0]
    val_rmse = rmse((a["pred"] - a["target"]) ** 2)
    surge_gain = gains[
        (gains.group == "all") & (gains.condition == "surge")
    ].image_gain_km_s
    quiet_gain = gains[
        (gains.group == "all") & (gains.condition == "quiet")
    ].image_gain_km_s
    print(f"(1) slow-fast launch COM gap : {m.slow_minus_fast_com_h:+.1f}h "
          f"(목표 >=20, 이론 {m.theory_gap_h:+.1f}, v6b ~7)")
    print(f"(2) rotation tracking slope  : "
          f"{m.rotation_slope_deg_per_h:+.3f} deg/h (이론 -0.550)")
    print(f"(3) best epoch / train gap   : history.csv와 --with-train으로 판정")
    print(f"(4) val RMSE                 : {val_rmse:.3f} (목표 <=67.5) / "
          f"surge gain {float(surge_gain.iloc[0]):+.1f}, "
          f"quiet gain {float(quiet_gain.iloc[0]):+.1f}")
    print(f"(5) gate {m.gate_mean:.2f} / coverage "
          f"{m.coverage_future_mean:.2f} / align KL {m.align_kl:.3f}")
    print(f"\nsaved to: {ANALYSIS_DIR.resolve()}")
