"""v5a+ 출력 성분 분해: 이미지 gain의 소유권 판정.

배경. decoder cross-attention의 image COM은 속도 그룹과 무관하게
~57h로 고정이다 -- 0..114h 균일 분포의 평균이 정확히 57h이므로,
decoder는 시점 선택 없이 이미지를 bag처럼 평균 낸다. 그러나 COM은
correction 경로만 보는 약한 도구다: encoder 혼합 후 attention 위치는
정보 출처를 말하지 않고, v5a에서 시간 정렬은 propagation branch의
소관이다. 문제는 best 체크포인트(epoch 4)의 propagation 동적 범위
(alpha x vstd ~ +-17 km/s)가 측정된 이미지 gain(+22~27)보다 작아서,
gain의 다수를 무정렬 correction이 운반했을 개연성이 있다는 것.
speed_ablation의 wind-only는 이미지를 통째로 꺼서 두 경로를 분리하지
못한다. 이 스크립트가 소유권을 성능 언어로 판정한다.

출력 분해 (model.py OUTPUT ASSEMBLY와 동일식):

    pred = base + alpha*(v_img - base) + correction

    base        : last_wind + beta*(clim - last_wind)   (anchor)
    base + prop : base + alpha*(v_img - base)           (정렬된 물리 경로)
    base + corr : base + correction                     (transformer 경로)
    full        : 전부

    prop own-gain = RMSE(base) - RMSE(base+prop)
    corr own-gain = RMSE(base) - RMSE(base+corr)
    synergy       = total gain - prop own - corr own

판정: prop own-gain이 surge/fast gain의 다수를 들면 "정렬이 작동해서
fast가 살아났다" 확정. corr own-gain이 지배하면 이미지는 무정렬 문맥
으로 쓰이는 것 -- bottleneck 강화(hindcast lambda 상향, correction
억제)로 이행.

추가로 사전 등록된 mechanism 진단:

    corr(s_t, 도착 시각의 실측 wind)

토큰별 속도 주장 s_t를, 그 토큰이 주장하는 도착 시각(tau_t - age_t)
의 실측 wind(과거는 관측 20점, 미래는 target 12점, 선형 보간)와
비교한다. gate 가중, 도착이 [-114h, +72h] 밖인 토큰은 제외. 양수로
크면 "각 이미지가 언제의 wind를 만드는지"를 실제로 읽는다는 증거.

사용법 (체크포인트/데이터가 있는 머신에서):

    python branch_decomposition.py

결과물: OUTPUT_DIR/branch_decomposition/ 아래 CSV + 콘솔 테이블.
"""

import numpy as np
import pandas as pd
import torch

from config import *
from dataset import val_loader
from inference import load_best_model

DECOMP_DIR = OUTPUT_DIR / "branch_decomposition"
DECOMP_DIR.mkdir(parents=True, exist_ok=True)

N_HORIZON = 12
STEP_HOURS = 6
SPEED_GROUP_LABELS = ("slow", "mid", "fast")
SURGE_THRESHOLD_KMS = 100.0
BOOTSTRAP_ITERS = 10_000

COMPONENT_LABELS = ("base", "base+prop", "base+corr", "full")


# ================================================================
# 1. Collection
# ================================================================

@torch.no_grad()
def collect(model, loader):
    """val 1 pass. 성분 일치 검증이 목적이므로 autocast 없이 fp32."""
    model.eval()
    keys = (
        "pred", "base", "v_img", "alpha", "correction",
        "target", "wind", "speed", "last_wind",
        "src_speed", "arrival", "gate",
    )
    out = {k: [] for k in keys}
    for batch in loader:
        images = batch["images"].to(DEVICE, non_blocking=PIN_MEMORY)
        wind = batch["wind"].to(DEVICE, non_blocking=PIN_MEMORY)
        pred = model(images, wind)
        out["pred"].append(pred.float().cpu().numpy())
        out["base"].append(model.last_base.float().cpu().numpy())
        out["v_img"].append(
            model.last_v_image_future.float().cpu().numpy()
        )
        out["alpha"].append(model.last_fusion_alpha.float().cpu().numpy())
        out["correction"].append(
            model.last_correction.float().cpu().numpy()
        )
        out["target"].append(batch["target"].numpy())
        out["wind"].append(batch["wind"].numpy())
        out["speed"].append(batch["wind"].mean(dim=1).numpy())
        out["last_wind"].append(batch["wind"][:, -1].numpy())
        out["src_speed"].append(
            model.last_source_speed_kms.float().cpu().numpy()
        )
        out["arrival"].append(
            model.last_arrival_hours.float().cpu().numpy()
        )
        out["gate"].append(model.last_source_gate.float().cpu().numpy())

    a = {k: np.concatenate(v).astype(np.float64) for k, v in out.items()}
    # /1000 스케일 -> km/s (alpha/gate 무단위, arrival은 시간,
    # src_speed는 model이 이미 km/s로 stash)
    for k in ("pred", "base", "v_img", "correction",
              "target", "wind", "speed", "last_wind"):
        a[k] = a[k] * 1000.0
    return a


# ================================================================
# 2. Metrics
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


# ================================================================
# 3. Main
# ================================================================

if __name__ == "__main__":
    model = load_best_model()
    if not model.use_images:
        raise SystemExit("use_images=False 모델은 분해 대상이 아님")

    a = collect(model, val_loader)
    n_samples = len(a["pred"])
    print(f"val samples: {n_samples}")

    # ---- 성분 재조립 및 sanity check ----
    predictions = {
        "base": a["base"],
        "base+prop": a["base"] + a["alpha"] * (a["v_img"] - a["base"]),
        "base+corr": a["base"] + a["correction"],
    }
    predictions["full"] = predictions["base+prop"] + a["correction"]
    reassembly_error = np.abs(predictions["full"] - a["pred"]).max()
    assert reassembly_error < 1.0, (
        f"재조립 불일치 {reassembly_error:.3f} km/s -- "
        "model.py stash와 조립식이 어긋남"
    )

    se = {
        name: (pred - a["target"]) ** 2
        for name, pred in predictions.items()
    }

    # ---- 그룹/조건 마스크 (speed_ablation과 동일 정의) ----
    speed_edges = np.quantile(a["speed"], [1 / 3, 2 / 3])
    group_ids = np.digitize(a["speed"], speed_edges)
    groups = {"all": np.ones(n_samples, dtype=bool)}
    for index, label in enumerate(SPEED_GROUP_LABELS):
        groups[label] = group_ids == index

    surge = a["target"].max(axis=1) - a["last_wind"]
    conditions = {
        "any": np.ones(n_samples, dtype=bool),
        "surge": surge > SURGE_THRESHOLD_KMS,
        "quiet": surge <= SURGE_THRESHOLD_KMS,
    }

    # ---- 분해 테이블 ----
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
            mse_base = se["base"][mask].mean(axis=1)
            gain_ci = {}
            for name in ("base+prop", "base+corr", "full"):
                mse_variant = se[name][mask].mean(axis=1)
                ci, p = paired_gain_bootstrap(mse_base, mse_variant)
                gain_ci[name] = (
                    rmse_by["base"] - rmse_by[name], ci, p
                )
            prop_own = gain_ci["base+prop"][0]
            corr_own = gain_ci["base+corr"][0]
            total = gain_ci["full"][0]
            rows.append({
                "group": group_label,
                "condition": condition_label,
                "count": n,
                **{f"rmse_{k}": v for k, v in rmse_by.items()},
                "prop_own_gain": prop_own,
                "prop_ci_low": gain_ci["base+prop"][1][0],
                "prop_ci_high": gain_ci["base+prop"][1][1],
                "prop_p_nonpositive": gain_ci["base+prop"][2],
                "corr_own_gain": corr_own,
                "corr_ci_low": gain_ci["base+corr"][1][0],
                "corr_ci_high": gain_ci["base+corr"][1][1],
                "corr_p_nonpositive": gain_ci["base+corr"][2],
                "total_gain": total,
                "synergy": total - prop_own - corr_own,
            })
    decomposition = pd.DataFrame(rows)
    decomposition.to_csv(DECOMP_DIR / "component_gain.csv", index=False)

    print("\n=== component RMSE (km/s) ===")
    pivot = decomposition[decomposition.condition == "any"].set_index(
        "group"
    )[[f"rmse_{k}" for k in COMPONENT_LABELS]]
    print(pivot.to_string(float_format=lambda v: f"{v:8.2f}"))

    print("\n=== own-gain vs base (km/s, paired bootstrap 95% CI) ===")
    for _, row in decomposition.iterrows():
        print(
            f"  {row.group:5s} {row.condition:6s} (n={int(row['count']):4d}): "
            f"prop {row.prop_own_gain:+7.2f} "
            f"[{row.prop_ci_low:+7.2f}, {row.prop_ci_high:+7.2f}]  "
            f"corr {row.corr_own_gain:+7.2f} "
            f"[{row.corr_ci_low:+7.2f}, {row.corr_ci_high:+7.2f}]  "
            f"total {row.total_gain:+7.2f}  synergy {row.synergy:+6.2f}"
        )

    # ---- horizon별 분해 (all 그룹) ----
    horizon_rows = []
    for j in range(N_HORIZON):
        row = {"horizon_hours": (j + 1) * STEP_HOURS}
        base_rmse = rmse(se["base"][:, j])
        for name in COMPONENT_LABELS:
            row[f"rmse_{name}"] = rmse(se[name][:, j])
        row["prop_own_gain"] = base_rmse - row["rmse_base+prop"]
        row["corr_own_gain"] = base_rmse - row["rmse_base+corr"]
        horizon_rows.append(row)
    by_horizon = pd.DataFrame(horizon_rows)
    by_horizon.to_csv(DECOMP_DIR / "gain_by_horizon.csv", index=False)
    print("\n=== own-gain by horizon (all, km/s) ===")
    print(by_horizon[
        ["horizon_hours", "prop_own_gain", "corr_own_gain"]
    ].to_string(index=False, float_format=lambda v: f"{v:7.2f}"))

    # ---- mechanism: corr(s_t, 도착 시각의 실측 wind) ----
    # 시간축: 관측 wind 20점 [-114..0] + target 12점 [+6..+72]
    time_grid = np.concatenate([
        np.arange(-19, 1) * float(STEP_HOURS),
        np.arange(1, 13) * float(STEP_HOURS),
    ])
    series = np.concatenate([a["wind"], a["target"]], axis=1)
    observed_at_arrival = np.stack([
        np.interp(a["arrival"][i], time_grid, series[i])
        for i in range(n_samples)
    ])
    in_window = (
        (a["arrival"] >= time_grid[0]) & (a["arrival"] <= time_grid[-1])
    )

    mech_rows = []
    for scope_label, scope in (
        ("all_tokens", in_window),
        ("future_arrivals", in_window & (a["arrival"] > 0)),
        ("past_arrivals", in_window & (a["arrival"] <= 0)),
    ):
        weights = (a["gate"] * scope).ravel()
        if weights.sum() <= 0:
            continue
        value = weighted_corr(
            a["src_speed"].ravel(),
            observed_at_arrival.ravel(),
            weights,
        )
        mech_rows.append({
            "scope": scope_label,
            "corr_src_speed_vs_observed": value,
            "token_weight_share": float(scope.mean()),
        })
    for group_label in SPEED_GROUP_LABELS:
        mask = groups[group_label][:, None] & in_window
        weights = (a["gate"] * mask).ravel()
        mech_rows.append({
            "scope": f"group_{group_label}",
            "corr_src_speed_vs_observed": weighted_corr(
                a["src_speed"].ravel(),
                observed_at_arrival.ravel(),
                weights,
            ),
            "token_weight_share": float(mask.mean()),
        })
    mechanism = pd.DataFrame(mech_rows)
    mechanism.to_csv(DECOMP_DIR / "mechanism.csv", index=False)
    print("\n=== mechanism: corr(s_t, wind@arrival), gate 가중 ===")
    print(mechanism.to_string(index=False,
                              float_format=lambda v: f"{v:7.3f}"))

    # ---- 레짐별 alpha / 전파 성분 진폭 ----
    prop_component = a["alpha"] * (a["v_img"] - a["base"])
    regime_rows = []
    for group_label, group_mask in groups.items():
        regime_rows.append({
            "group": group_label,
            "alpha_mean": float(a["alpha"][group_mask].mean()),
            "prop_component_std_kms":
                float(prop_component[group_mask].std()),
            "correction_std_kms":
                float(a["correction"][group_mask].std()),
            "v_img_std_kms": float(a["v_img"][group_mask].std()),
        })
    regime = pd.DataFrame(regime_rows)
    regime.to_csv(DECOMP_DIR / "regime_stats.csv", index=False)
    print("\n=== 레짐별 성분 통계 ===")
    print(regime.to_string(index=False,
                           float_format=lambda v: f"{v:8.2f}"))

    print(f"\nsaved to: {DECOMP_DIR.resolve()}")
