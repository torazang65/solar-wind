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

import numpy as np
import pandas as pd
import torch
from torch import nn

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import *
from dataset import val_loader
from inference import load_best_model

ANALYSIS_DIR = OUTPUT_DIR / "attention_analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

N_INPUT = 20        # input timesteps (images / wind)
N_HORIZON = 12      # future queries
STEP_HOURS = 6

# model.py 주석과 동일한 전파 시간 가정: 1 AU를 300~800 km/s로 통과
TRANSIT_MIN_H = 48.0
TRANSIT_MAX_H = 120.0

# 검증셋 전체가 부담스러우면 배치 수 제한 (None = 전체)
MAX_BATCHES = None


def hours_ago(index):
    """input timestep index (0=가장 오래됨, 19=가장 최근) -> 관측시점 기준 몇 시간 전인지."""
    return (N_INPUT - 1 - index) * STEP_HOURS


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
def collect_mean_attention(model, loader):
    """검증셋 전체에 대해 attention weight의 샘플 평균을 계산한다.

    returns: dict[name] -> (nhead, L, S) float64, 그리고 샘플 수
    """
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)

    model.eval()
    recorder = AttentionRecorder(model)

    sums, count = {}, 0
    for batch_index, batch in enumerate(loader):
        if MAX_BATCHES is not None and batch_index >= MAX_BATCHES:
            break
        images = batch["images"].to(DEVICE, non_blocking=PIN_MEMORY)
        wind = batch["wind"].to(DEVICE, non_blocking=PIN_MEMORY)

        recorder.store.clear()
        model(images, wind)

        for name, weights in recorder.store.items():
            batch_sum = weights.numpy().astype(np.float64).sum(axis=0)
            if name not in sums:
                sums[name] = np.zeros_like(batch_sum)
            sums[name] += batch_sum
        count += wind.size(0)
        print(f"batch {batch_index + 1}: {count} samples", flush=True)

    return {name: total / count for name, total in sums.items()}, count


# ================================================================
# 2. Theory band helpers
# ================================================================

def decoder_theory_band(horizon_hours):
    """horizon h에 도착하는 태양풍의 태양 출발 시각을 이미지 나이(시간 전)로.

    출발 시각 = (t0 + h) - transit, transit in [48, 120]h
    -> 이미지 나이 a = transit - h, 관측창 [0, 114]h로 클리핑
    """
    low = max(0.0, TRANSIT_MIN_H - horizon_hours)
    high = min(hours_ago(0), TRANSIT_MAX_H - horizon_hours)
    return low, high


def center_of_mass_hours(attention_row):
    """이미지 토큰 20개에 대한 attention 분포의 무게중심 (hours ago)."""
    ages = np.array([hours_ago(i) for i in range(N_INPUT)], dtype=np.float64)
    mass = attention_row.sum()
    if mass <= 0:
        return np.nan
    return float((attention_row * ages).sum() / mass)


def spearman(x, y):
    rank_x = np.argsort(np.argsort(x)).astype(np.float64)
    rank_y = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


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

    # 이론 밴드: 각 horizon이 봐야 할 이미지 나이 구간을 이미지 패널에 표시
    for j in range(N_HORIZON):
        low, high = decoder_theory_band((j + 1) * STEP_HOURS)
        if low > high:
            continue
        # hours ago -> index
        x_low = N_INPUT - 1 - high / STEP_HOURS
        x_high = N_INPUT - 1 - low / STEP_HOURS
        axes[0].plot([x_low, x_high], [j, j], color="red", lw=1.2, alpha=0.8)
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
    com = np.array([
        center_of_mass_hours(cross_mean[j, :N_INPUT]) for j in range(N_HORIZON)
    ])
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


# ================================================================
# 4. Main
# ================================================================

if __name__ == "__main__":
    model = load_best_model()
    if not model.use_images:
        raise SystemExit("use_images=False 모델은 이미지 attention 분석 대상이 아님")

    mean_attention, sample_count = collect_mean_attention(model, val_loader)
    print(f"\naveraged over {sample_count} validation samples")
    for name, mean in mean_attention.items():
        print(f"  {name}: {mean.shape}")

    # 원본 행렬 저장 (추가 분석용)
    np.savez(
        ANALYSIS_DIR / "attention_mean.npz",
        **{name.replace(".", "_"): mean for name, mean in mean_attention.items()},
        sample_count=sample_count,
    )

    # ---- decoder cross-attention ----
    cross_name = next(
        name for name in mean_attention if name.endswith("multihead_attn")
    )
    cross_heads = mean_attention[cross_name]        # (nhead, 12, 40)
    cross_mean = cross_heads.mean(axis=0)           # (12, 40)

    plot_decoder_cross(cross_mean)
    plot_decoder_cross_heads(cross_heads)
    com = plot_image_com(cross_mean)
    image_share, wind_share = plot_modality_share(cross_mean)

    # ---- encoder self-attention ----
    encoder_means = sorted(
        (name, heads.mean(axis=0))
        for name, heads in mean_attention.items()
        if "transformer_encoder" in name
    )
    plot_encoder_layers(encoder_means)
    plot_encoder_wind_to_image(encoder_means)

    # ---- summary table ----
    horizons = np.array([(j + 1) * STEP_HOURS for j in range(N_HORIZON)])
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
    print(f"\nsaved plots to: {ANALYSIS_DIR.resolve()}")
