"""Inference-time attention analysis.

물리 가설 검증용 스크립트:

  지구에서 관측되는 태양풍은 태양을 떠난 지 2~5일(48~120h, 6h 스텝으로
  8~20 스텝) 뒤에 도착한다. 따라서

  1. 디코더 cross-attention: +6h처럼 가까운 horizon 쿼리는 "오래된"
     이미지 토큰을, +72h처럼 먼 horizon 쿼리는 "최근" 이미지 토큰을
     더 봐야 한다. (도착시각 - 통과시간 = 태양 출발 시각이 이미지
     윈도우 안에서 horizon이 멀수록 최근 쪽으로 이동)
  2. 인코더 self-attention의 wind->image 블록: wind_t 토큰은 자기보다
     8~20 스텝 이전의 이미지 토큰에 attention 해야 한다.
  3. 속도 조건부: 통과시간은 풍속에 반비례하므로, 관측 풍속이 빠른
     샘플일수록 attention이 "최근" 이미지 쪽으로 이동해야 한다.
     검증셋을 입력 wind 평균으로 3분위(slow/mid/fast) 나눠 그룹별
     COM을 각 그룹 평균 풍속에서 유도한 이론 곡선과 비교한다.

사용법 (체크포인트/데이터가 있는 머신에서):

    python attention_analysis.py

결과물은 OUTPUT_DIR/attention_analysis/ 아래에 PNG + CSV + NPZ로 저장.

구현 노트: nn.TransformerEncoderLayer는 self_attn을 need_weights=False로
호출하고, eval+no_grad에서는 fastpath(C++ 커널)를 타서 self_attn.forward
자체가 호출되지 않을 수 있다. 그래서 (a) fastpath를 전역으로 끄고
(b) MultiheadAttention.forward를 감싸 need_weights=True,
average_attn_weights=False로 강제해 per-head weight를 뽑는다.
AMP는 쓰지 않는다(분석은 1회 pass라 fp32로 충분).
"""

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import *
from dataset import val_loader, val_dataset
from inference import load_best_model

ANALYSIS_DIR = OUTPUT_DIR / "attention_analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

N_INPUT = 20        # input timesteps (images / wind)
N_HORIZON = 12      # future queries
STEP_HOURS = 6

# model.py 주석과 동일한 전파 시간 가정: 1 AU를 300~800 km/s로 통과
TRANSIT_MIN_H = 48.0
TRANSIT_MAX_H = 120.0
AU_KM = 1.496e8

# 속도 그룹 (입력 wind 20개의 평균, 3분위 분할)
SPEED_GROUP_LABELS = ("slow", "mid", "fast")
SPEED_GROUP_COLORS = ("tab:blue", "tab:green", "tab:orange")

# 검증셋 전체가 부담스러우면 배치 수 제한 (None = 전체)
MAX_BATCHES = None


def hours_ago(index):
    """input timestep index (0=가장 오래됨, 19=가장 최근) -> 관측시점 기준 몇 시간 전인지."""
    return (N_INPUT - 1 - index) * STEP_HOURS


def transit_hours(speed_kms):
    """풍속(km/s) -> 1 AU 통과 시간(hours)."""
    return AU_KM / speed_kms / 3600.0


# ================================================================
# 1. Attention capture
# ================================================================

class AttentionRecorder:
    """모델 안의 모든 nn.MultiheadAttention을 패치해 마지막 forward의
    per-head attention weight를 module 이름으로 저장한다."""

    def __init__(self, model):
        self.store = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                self._patch(name, module)

    def _patch(self, name, module):
        original = module.forward

        def wrapped(*args, __name=name, __original=original, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            output, weights = __original(*args, **kwargs)
            # weights: (B, nhead, L, S)
            self.store[__name] = weights.detach().float().cpu()
            return output, weights

        module.forward = wrapped


@torch.no_grad()
def collect_mean_attention(model, loader, speed_edges=None):
    """검증셋에 대해 attention weight의 샘플 평균을 (속도 그룹별로) 계산.

    speed_edges: 오름차순 분위 경계 (km/s) 2개. None이면 단일 그룹 "all".

    returns: dict[label] -> {
        "means": dict[name] -> (nhead, L, S) float64,
        "count": int,
        "mean_speed_kms": float,
    }
    """
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)

    model.eval()
    recorder = AttentionRecorder(model)

    if speed_edges is None:
        labels = ("all",)
    else:
        labels = SPEED_GROUP_LABELS

    sums = defaultdict(dict)          # label -> name -> array sum
    counts = defaultdict(int)         # label -> n samples
    speed_totals = defaultdict(float)  # label -> speed sum (km/s)

    total = 0
    for batch_index, batch in enumerate(loader):
        if MAX_BATCHES is not None and batch_index >= MAX_BATCHES:
            break
        images = batch["images"].to(DEVICE, non_blocking=PIN_MEMORY)
        wind = batch["wind"].to(DEVICE, non_blocking=PIN_MEMORY)

        # dataset.py에서 /1000된 값이므로 km/s로 복원
        speeds = wind.mean(dim=1).double().cpu().numpy() * 1000.0
        if speed_edges is None:
            group_ids = np.zeros(len(speeds), dtype=np.int64)
        else:
            group_ids = np.digitize(speeds, speed_edges)

        recorder.store.clear()
        model(images, wind)

        for name, weights in recorder.store.items():
            batch_weights = weights.numpy().astype(np.float64)  # (B,nh,L,S)
            for group in np.unique(group_ids):
                label = labels[group]
                group_sum = batch_weights[group_ids == group].sum(axis=0)
                if name not in sums[label]:
                    sums[label][name] = np.zeros_like(group_sum)
                sums[label][name] += group_sum

        for group in np.unique(group_ids):
            label = labels[group]
            mask = group_ids == group
            counts[label] += int(mask.sum())
            speed_totals[label] += float(speeds[mask].sum())

        total += wind.size(0)
        print(f"batch {batch_index + 1}: {total} samples", flush=True)

    return {
        label: {
            "means": {
                name: array / counts[label]
                for name, array in sums[label].items()
            },
            "count": counts[label],
            "mean_speed_kms": speed_totals[label] / counts[label],
        }
        for label in labels
        if counts[label] > 0
    }


def combine_groups(group_results):
    """그룹별 평균을 샘플 수 가중으로 합쳐 전체 평균을 복원한다."""
    total = sum(g["count"] for g in group_results.values())
    names = next(iter(group_results.values()))["means"].keys()
    combined = {}
    for name in names:
        combined[name] = sum(
            g["means"][name] * g["count"] for g in group_results.values()
        ) / total
    return combined, total


# ================================================================
# 2. Theory / metric helpers
# ================================================================

def decoder_theory_band(horizon_hours):
    """horizon h에 도착하는 태양풍의 태양 출발 시각을 이미지 나이(시간 전)로.

    출발 시각 = (t0 + h) - transit, transit in [48, 120]h
    -> 이미지 나이 a = transit - h, 관측창 [0, 114]h로 클리핑
    """
    low = max(0.0, TRANSIT_MIN_H - horizon_hours)
    high = min(hours_ago(0), TRANSIT_MAX_H - horizon_hours)
    return low, high


def theory_com_curve(speed_kms, horizons):
    """단일 풍속 가정에서 horizon별 '봐야 할' 이미지 나이 (클리핑 포함)."""
    transit = transit_hours(speed_kms)
    return np.clip(transit - horizons, 0.0, hours_ago(0))


def center_of_mass_hours(attention_row):
    """이미지 토큰 20개에 대한 attention 분포의 무게중심 (hours ago)."""
    ages = np.array([hours_ago(i) for i in range(N_INPUT)], dtype=np.float64)
    mass = attention_row.sum()
    if mass <= 0:
        return np.nan
    return float((attention_row * ages).sum() / mass)


def image_com_per_horizon(cross_mean):
    return np.array([
        center_of_mass_hours(cross_mean[j, :N_INPUT])
        for j in range(N_HORIZON)
    ])


def spearman(x, y):
    rank_x = np.argsort(np.argsort(x)).astype(np.float64)
    rank_y = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def lag_residual_stats(block):
    """인코더 wind(query)->image(key) 블록의 lag 구조 요약.

    column preference(모든 wind 토큰이 공통으로 보는 이미지 슬롯)를
    제거한 잔차를 lag별로 평균 낸다. 0이면 lag 선호 없음.
    returns: (band[8,20), causal[1,7], anticausal[<0]) 잔차 평균
    """
    dist = block / block.sum(axis=1, keepdims=True)
    resid = dist - dist.mean(axis=0, keepdims=True)
    by_lag = defaultdict(list)
    for t in range(N_INPUT):
        for s in range(N_INPUT):
            by_lag[t - s].append(resid[t, s])
    profile = {lag: float(np.mean(v)) for lag, v in by_lag.items()}
    band = np.mean([profile[k] for k in range(8, N_INPUT) if k in profile])
    causal = np.mean([profile[k] for k in range(1, 8)])
    anti = np.mean([profile[k] for k in profile if k < 0])
    return float(band), float(causal), float(anti)


# ================================================================
# 3. Plots
# ================================================================

def _time_ticks(ax, axis="x"):
    ticks = list(range(0, N_INPUT, 2))
    labels = [f"-{hours_ago(i)}h" for i in ticks]
    if axis == "x":
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45, fontsize=7)
    else:
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=7)


def _horizon_ticks(ax):
    ax.set_yticks(range(N_HORIZON))
    ax.set_yticklabels(
        [f"+{(j + 1) * STEP_HOURS}h" for j in range(N_HORIZON)], fontsize=7
    )


def _draw_theory_band_rows(ax):
    for j in range(N_HORIZON):
        low, high = decoder_theory_band((j + 1) * STEP_HOURS)
        if low > high:
            continue
        x_low = N_INPUT - 1 - high / STEP_HOURS
        x_high = N_INPUT - 1 - low / STEP_HOURS
        ax.plot([x_low, x_high], [j, j], color="red", lw=1.2, alpha=0.8)


def plot_decoder_cross(cross_mean):
    """cross_mean: (12, 40) head-averaged. 이미지/wind 절반을 나란히."""
    image_part = cross_mean[:, :N_INPUT]
    wind_part = cross_mean[:, N_INPUT:]
    vmax = cross_mean.max()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for ax, part, title in (
        (axes[0], image_part, "decoder cross-attn -> IMAGE tokens"),
        (axes[1], wind_part, "decoder cross-attn -> WIND tokens"),
    ):
        im = ax.imshow(part, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("input time (hours before t0)")
        ax.set_ylabel("forecast horizon")
        _time_ticks(ax)
        _horizon_ticks(ax)
        fig.colorbar(im, ax=ax, shrink=0.85)

    _draw_theory_band_rows(axes[0])
    axes[0].plot([], [], color="red", lw=1.2, label="theory (48-120h transit)")
    axes[0].legend(loc="upper right", fontsize=7)

    fig.savefig(ANALYSIS_DIR / "decoder_cross_attention.png", dpi=150)
    plt.close(fig)


def plot_decoder_cross_heads(cross_heads):
    """cross_heads: (nhead, 12, 40). 이미지 절반만 head별 그리드."""
    nhead = cross_heads.shape[0]
    ncols = 4
    nrows = int(np.ceil(nhead / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), constrained_layout=True
    )
    vmax = cross_heads[:, :, :N_INPUT].max()
    for head in range(nrows * ncols):
        ax = axes.flat[head]
        if head >= nhead:
            ax.axis("off")
            continue
        ax.imshow(
            cross_heads[head, :, :N_INPUT],
            aspect="auto", cmap="viridis", vmin=0, vmax=vmax,
        )
        ax.set_title(f"head {head}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("decoder cross-attn -> IMAGE tokens, per head", fontsize=11)
    fig.savefig(ANALYSIS_DIR / "decoder_cross_attention_heads.png", dpi=150)
    plt.close(fig)


def plot_image_com(cross_mean):
    """horizon별 이미지 attention 무게중심 vs 이론 밴드."""
    horizons = np.array([(j + 1) * STEP_HOURS for j in range(N_HORIZON)])
    com = image_com_per_horizon(cross_mean)
    band = np.array([decoder_theory_band(h) for h in horizons])
    band_low = np.minimum(band[:, 0], band[:, 1])
    band_high = np.maximum(band[:, 0], band[:, 1])

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.fill_between(
        horizons, band_low, band_high,
        color="red", alpha=0.15, label="theory (48-120h transit)",
    )
    ax.plot(horizons, com, "o-", color="tab:blue", label="attention center of mass")
    ax.set_xlabel("forecast horizon (hours)")
    ax.set_ylabel("image age (hours before t0)")
    ax.set_title("which image age does each horizon attend to?")
    ax.invert_yaxis()  # 위 = 최근 이미지
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(ANALYSIS_DIR / "decoder_image_com.png", dpi=150)
    plt.close(fig)
    return com


def plot_modality_share(cross_mean):
    """horizon별 image vs wind attention 총량."""
    image_share = cross_mean[:, :N_INPUT].sum(axis=1)
    wind_share = cross_mean[:, N_INPUT:].sum(axis=1)
    horizons = [f"+{(j + 1) * STEP_HOURS}" for j in range(N_HORIZON)]

    fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    x = np.arange(N_HORIZON)
    ax.bar(x - 0.2, image_share, width=0.4, label="image tokens")
    ax.bar(x + 0.2, wind_share, width=0.4, label="wind tokens")
    ax.set_xticks(x)
    ax.set_xticklabels(horizons, fontsize=8)
    ax.set_xlabel("forecast horizon (hours)")
    ax.set_ylabel("total attention mass")
    ax.set_title("decoder cross-attn: image vs wind share per horizon")
    ax.legend(fontsize=8)
    fig.savefig(ANALYSIS_DIR / "decoder_modality_share.png", dpi=150)
    plt.close(fig)
    return image_share, wind_share


def plot_encoder_layers(encoder_means):
    """각 인코더 레이어의 40x40 head-averaged self-attention."""
    n_layers = len(encoder_means)
    fig, axes = plt.subplots(
        1, n_layers, figsize=(5.5 * n_layers, 4.8), constrained_layout=True
    )
    if n_layers == 1:
        axes = [axes]
    for layer_index, (name, mean) in enumerate(encoder_means):
        ax = axes[layer_index]
        im = ax.imshow(mean, aspect="auto", cmap="viridis")
        # 모달리티 경계선 (0-19: image, 20-39: wind)
        ax.axhline(N_INPUT - 0.5, color="white", lw=0.8)
        ax.axvline(N_INPUT - 0.5, color="white", lw=0.8)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("key token (0-19 img, 20-39 wind)")
        ax.set_ylabel("query token")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(ANALYSIS_DIR / "encoder_self_attention.png", dpi=150)
    plt.close(fig)


def plot_encoder_wind_to_image(encoder_means):
    """wind(query) -> image(key) 블록과 이론 대각 밴드 (lag 8~20 스텝)."""
    n_layers = len(encoder_means)
    fig, axes = plt.subplots(
        1, n_layers, figsize=(5.5 * n_layers, 4.8), constrained_layout=True
    )
    if n_layers == 1:
        axes = [axes]
    for layer_index, (name, mean) in enumerate(encoder_means):
        block = mean[N_INPUT:, :N_INPUT]  # rows: wind_t, cols: image_s
        ax = axes[layer_index]
        im = ax.imshow(block, aspect="auto", cmap="viridis")
        # 이론: image index s in [t-20, t-8]
        t = np.arange(N_INPUT)
        ax.plot(t - 8, t, color="red", lw=1.0, label="lag 8 steps (48h)")
        ax.plot(t - 20, t, color="orange", lw=1.0, label="lag 20 steps (120h)")
        ax.plot(t, t, color="white", lw=0.8, ls="--", label="same timestamp")
        ax.set_xlim(-0.5, N_INPUT - 0.5)
        ax.set_ylim(N_INPUT - 0.5, -0.5)
        ax.set_title(f"{name}\nwind(query) -> image(key)", fontsize=9)
        ax.set_xlabel("image timestep")
        ax.set_ylabel("wind timestep")
        if layer_index == 0:
            ax.legend(fontsize=7, loc="lower right")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(ANALYSIS_DIR / "encoder_wind_to_image.png", dpi=150)
    plt.close(fig)


def plot_image_com_by_speed(group_results, cross_name):
    """속도 그룹별 이미지 COM vs 각 그룹 평균 풍속의 이론 곡선.

    가설: 빠른 바람 그룹일수록 COM이 최근(작은 나이) 쪽에 있어야 한다.
    """
    horizons = np.arange(1, N_HORIZON + 1) * STEP_HOURS

    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    for label, color in zip(SPEED_GROUP_LABELS, SPEED_GROUP_COLORS):
        if label not in group_results:
            continue
        group = group_results[label]
        cross_mean = group["means"][cross_name].mean(axis=0)
        com = image_com_per_horizon(cross_mean)
        speed = group["mean_speed_kms"]
        ax.plot(
            horizons, com, "o-", color=color,
            label=f"{label} (v={speed:.0f} km/s, n={group['count']})",
        )
        ax.plot(
            horizons, theory_com_curve(speed, horizons),
            ls="--", color=color, alpha=0.6,
            label=f"{label} theory (transit {transit_hours(speed):.0f}h)",
        )
    ax.set_xlabel("forecast horizon (hours)")
    ax.set_ylabel("image age (hours before t0)")
    ax.set_title("image attention COM by observed wind speed\n"
                 "(faster wind -> shorter transit -> more recent images?)")
    ax.invert_yaxis()
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.savefig(ANALYSIS_DIR / "decoder_image_com_by_speed.png", dpi=150)
    plt.close(fig)


def plot_cross_by_speed(group_results, cross_name):
    """속도 그룹별 디코더->이미지 히트맵 (공유 컬러스케일)."""
    present = [l for l in SPEED_GROUP_LABELS if l in group_results]
    vmax = max(
        group_results[l]["means"][cross_name].mean(axis=0)[:, :N_INPUT].max()
        for l in present
    )
    fig, axes = plt.subplots(
        1, len(present), figsize=(4.6 * len(present), 4.2),
        constrained_layout=True,
    )
    if len(present) == 1:
        axes = [axes]
    for ax, label in zip(axes, present):
        group = group_results[label]
        image_part = group["means"][cross_name].mean(axis=0)[:, :N_INPUT]
        im = ax.imshow(
            image_part, aspect="auto", cmap="viridis", vmin=0, vmax=vmax
        )
        _draw_theory_band_rows(ax)
        ax.set_title(
            f"{label} (v={group['mean_speed_kms']:.0f} km/s)", fontsize=9
        )
        ax.set_xlabel("input time")
        _time_ticks(ax)
        _horizon_ticks(ax)
    fig.colorbar(im, ax=axes[-1], shrink=0.85)
    fig.suptitle("decoder cross-attn -> IMAGE tokens by wind speed", fontsize=11)
    fig.savefig(ANALYSIS_DIR / "decoder_cross_attention_by_speed.png", dpi=150)
    plt.close(fig)


# ================================================================
# 4. Main
# ================================================================

if __name__ == "__main__":
    model = load_best_model()
    if not model.use_images:
        raise SystemExit("use_images=False 모델은 이미지 attention 분석 대상이 아님")
    if not getattr(model, "use_correction", True):
        raise SystemExit(
            "use_correction=False(v6a) 모델은 encoder/decoder가 없어 "
            "attention 분석 대상이 아님. 시간 정렬 진단은 "
            "branch_decomposition.py를 돌린 뒤 propagation_alignment.py"
            "(components.npz 기반, 로컬 실행 가능)가 대체한다"
        )

    # 속도 3분위 경계 (검증셋 입력 wind 평균, km/s)
    val_speed_kms = val_dataset.wind.mean(axis=1) * 1000.0
    speed_edges = np.quantile(val_speed_kms, [1 / 3, 2 / 3])
    print(
        f"speed tercile edges: {speed_edges[0]:.0f} / {speed_edges[1]:.0f} km/s"
    )

    group_results = collect_mean_attention(
        model, val_loader, speed_edges=speed_edges
    )
    mean_attention, sample_count = combine_groups(group_results)
    print(f"\naveraged over {sample_count} validation samples")
    for label, group in group_results.items():
        print(
            f"  {label}: n={group['count']}, "
            f"mean speed {group['mean_speed_kms']:.0f} km/s "
            f"(transit ~{transit_hours(group['mean_speed_kms']):.0f}h)"
        )

    # 원본 행렬 저장 (전체 + 그룹별, 추가 분석용)
    npz_payload = {
        name.replace(".", "_"): mean for name, mean in mean_attention.items()
    }
    for label, group in group_results.items():
        for name, mean in group["means"].items():
            npz_payload[f"{label}__{name.replace('.', '_')}"] = mean
        npz_payload[f"{label}__count"] = group["count"]
        npz_payload[f"{label}__mean_speed_kms"] = group["mean_speed_kms"]
    np.savez(
        ANALYSIS_DIR / "attention_mean.npz",
        **npz_payload,
        sample_count=sample_count,
    )

    # ---- decoder cross-attention (전체 평균) ----
    cross_name = next(
        name for name in mean_attention if name.endswith("multihead_attn")
    )
    cross_heads = mean_attention[cross_name]        # (nhead, 12, 40)
    cross_mean = cross_heads.mean(axis=0)           # (12, 40)

    plot_decoder_cross(cross_mean)
    plot_decoder_cross_heads(cross_heads)
    com = plot_image_com(cross_mean)
    image_share, wind_share = plot_modality_share(cross_mean)

    # ---- encoder self-attention (전체 평균) ----
    encoder_means = sorted(
        (name, heads.mean(axis=0))
        for name, heads in mean_attention.items()
        if "transformer_encoder" in name
    )
    plot_encoder_layers(encoder_means)
    plot_encoder_wind_to_image(encoder_means)

    # ---- 속도 조건부 ----
    plot_image_com_by_speed(group_results, cross_name)
    plot_cross_by_speed(group_results, cross_name)

    horizons = np.array([(j + 1) * STEP_HOURS for j in range(N_HORIZON)])
    speed_rows = []
    for label in SPEED_GROUP_LABELS:
        if label not in group_results:
            continue
        group = group_results[label]
        group_cross = group["means"][cross_name].mean(axis=0)
        group_com = image_com_per_horizon(group_cross)
        group_speed = group["mean_speed_kms"]
        theory = theory_com_curve(group_speed, horizons)
        for j in range(N_HORIZON):
            speed_rows.append({
                "group": label,
                "mean_speed_kms": group_speed,
                "count": group["count"],
                "horizon_hours": horizons[j],
                "image_com_hours_ago": group_com[j],
                "theory_com_hours_ago": theory[j],
                "image_attention_share": group_cross[j, :N_INPUT].sum(),
            })
    speed_summary = pd.DataFrame(speed_rows)
    speed_summary.to_csv(ANALYSIS_DIR / "speed_summary.csv", index=False)

    # ---- summary table (전체 평균) ----
    band = np.array([decoder_theory_band(h) for h in horizons])
    summary = pd.DataFrame({
        "horizon_hours": horizons,
        "image_attention_share": image_share,
        "wind_attention_share": wind_share,
        "image_com_hours_ago": com,
        "theory_low_hours_ago": np.minimum(band[:, 0], band[:, 1]),
        "theory_high_hours_ago": np.maximum(band[:, 0], band[:, 1]),
    })
    summary.to_csv(ANALYSIS_DIR / "summary.csv", index=False)

    rho = spearman(horizons.astype(np.float64), com)
    print("\n=== decoder cross-attention summary ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(
        f"\nSpearman(horizon, image attention COM) = {rho:.3f}"
        "  (이론대로면 음수: horizon이 멀수록 최근 이미지)"
    )
    in_band = (
        (com >= summary.theory_low_hours_ago.to_numpy())
        & (com <= summary.theory_high_hours_ago.to_numpy())
    )
    print(f"COM inside theory band: {int(in_band.sum())}/{N_HORIZON} horizons")

    # ---- 속도 조건부 요약 ----
    print("\n=== speed-conditioned image COM ===")
    pivot = speed_summary.pivot(
        index="horizon_hours", columns="group", values="image_com_hours_ago"
    )
    pivot = pivot[[l for l in SPEED_GROUP_LABELS if l in pivot.columns]]
    print(pivot.to_string(float_format=lambda v: f"{v:.1f}"))
    if {"slow", "fast"}.issubset(pivot.columns):
        gap = pivot["slow"] - pivot["fast"]
        print(
            f"\nslow - fast COM gap: mean {gap.mean():+.1f}h "
            f"(이론대로면 양수: 빠른 바람일수록 최근 이미지), "
            f"positive at {(gap > 0).sum()}/{N_HORIZON} horizons"
        )

    # ---- 속도 그룹별 인코더 lag 구조 ----
    print("\n=== encoder wind->image lag residual by speed group ===")
    print("(column preference 제거 후 lag별 잔차 평균; 0이면 lag 선호 없음)")
    for label in SPEED_GROUP_LABELS:
        if label not in group_results:
            continue
        for name in sorted(group_results[label]["means"]):
            if "transformer_encoder" not in name:
                continue
            block = group_results[label]["means"][name].mean(axis=0)[
                N_INPUT:, :N_INPUT
            ]
            band_r, causal_r, anti_r = lag_residual_stats(block)
            print(
                f"  {label:5s} {name}: band[8,20]={band_r:+.5f} "
                f"causal[1,7]={causal_r:+.5f} anticausal={anti_r:+.5f}"
            )

    print(f"\nsaved plots to: {ANALYSIS_DIR.resolve()}")
