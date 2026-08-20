import os
import time
import math
import numpy as np
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

# v5a: 물리 스칼라 전용 param group. AdamW의 스텝 크기는 gradient
# 스케일과 무관하게 ~lr이라, 총 이동 예산이 sum(lr) ~= 0.1 수준이다.
# 자연 스케일이 O(1)~O(10)인 스칼라(reversion logit -4->0, dist_eff raw
# 등)는 base lr로는 초기값에 얼어붙는다 -- v2/v3
# prior gate(-4)가 유인과 무관하게 기계적으로 못 열린 원인이기도
# 하다 (v3의 tau 이동 관측치 ~0.4h가 이 예산과 정량 일치). 100x lr,
# weight decay 0 (decay가 물리 상수를 0으로 끌어당기지 않도록).
def is_physical_param(name):
    return (
        name in {
            "dist_eff_raw", "reversion_logit",
            "climatology", "fallback_weight_raw",
        }
        or name in {
            "source_speed_head.bias", "source_gate_head.bias",
            "transit_residual_head.bias", "lon_offset_head.bias",
        }
    )

physical_params = [
    p for n, p in model.named_parameters() if is_physical_param(n)
]
base_params = [
    p for n, p in model.named_parameters() if not is_physical_param(n)
]
optimizer = torch.optim.AdamW(
    [
        {"params": base_params, "lr": PEAK_LR, "weight_decay": 0.01},
        {
            "params": physical_params,
            "lr": PEAK_LR * PHYSICAL_LR_MULT,
            "weight_decay": 0.0,
        },
    ]
)

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

# v5b surge BCE의 클래스 불균형 보정 (val 기준 surge:quiet ~ 363:836).
surge_pos_weight = torch.tensor(SURGE_POS_WEIGHT, device=DEVICE)
checkpoint_path = RUN_DIR / "best_model.pth"


def alignment_kl(model, aux, wind, target):
    """v7 backmapping alignment: label 역산 소스 궤적 vs kernel weight.

    config.py의 ALIGN_* 주석이 설계 근거. 여기는 조립만:

      y_series (B,25) = 관측 wind 13점(-72..0h) + target 12점(+6..+72h)
      tau* = D_eff.detach() / y            radial 통과시간 역산
      lon*(t,u) = -omega*(u - tau* + age_t)  각 프레임에서의 소스 경도
      q (B,20,2,4,25): lon* 주변 가우시안, 위도 2행 균등 분할, 가시
        원반 밖(|lon*|>90)은 질량 0. 소스 축 정규화.
      w~: aux["source_weight"] 정규화본.
      KL(q || w~)를 유효 (sample, u)에 평균 -- modality drop 샘플과
      전 프레임이 원반 밖인 u는 제외.

    fp32 강제: kernel exp/정규화가 fp16에서 underflow하면 log(0)이 난다.
    returns: (kl_mean, valid_share) -- 둘 다 스칼라 텐서/float.
    """
    with torch.autocast(DEVICE.type, enabled=False):
        w = aux["source_weight"].float()
        batch_size = w.size(0)
        dist_eff = (
            30.0 + 25.0 * torch.sigmoid(model.dist_eff_raw)
        ).detach()
        time_grid = torch.cat(
            [model.hindcast_hours, model.horizon_hours]
        )  # (25,)
        ages = model.image_age_hours  # (20,)
        omega = model.omega_deg_per_hour

        y_series = torch.cat([wind[:, 7:], target], dim=1).float()
        # [200,1200] km/s 클램프: 나눗셈 발산 방지 (라벨 이상치 방어).
        tau_star = dist_eff / y_series.clamp(0.2, 1.2)  # (B,25)

        # (B,20,25): 프레임 -> 출발까지의 시간차, 되감은 경도.
        delta_t = (
            time_grid.view(1, 1, -1)
            - tau_star.unsqueeze(1)
            + ages.view(1, -1, 1)
        )
        lon_star = -omega * delta_t
        visible = lon_star.abs() <= 90.0

        # (B,20,4,25) 경도 가우시안 -> 위도 2행 균등 -> (B,20,2,4,25)
        gaussian = torch.exp(
            -((model.cell_lon_deg.view(1, 1, 4, 1)
               - lon_star.unsqueeze(2)) ** 2)
            / (2.0 * ALIGN_SIGMA_DEG ** 2)
        ) * visible.unsqueeze(2)
        q = gaussian.unsqueeze(2).expand(-1, -1, 2, -1, -1) / 2.0
        q_mass = q.sum(dim=(1, 2, 3))  # (B,25)
        valid = (q_mass > 1e-6).float() * aux[
            "image_keep"
        ].float().unsqueeze(-1)
        q = q / q_mass.clamp_min(1e-6).view(batch_size, 1, 1, 1, -1)

        w = w + 1e-8
        w = w / w.sum(dim=(1, 2, 3), keepdim=True)
        kl = (q * (q.clamp_min(1e-12).log() - w.log())).sum(
            dim=(1, 2, 3)
        )  # (B,25)
        kl_mean = (kl * valid).sum() / valid.sum().clamp(min=1.0)
        return kl_mean, float(valid.mean())

if checkpoint_path.exists():
    checkpoint_path.unlink()

# v6b: EPOCHS(35)의 cosine tail을 관측할 수 있도록 patience도 config와
# 함께 관리한다. v6a의 20은 100-epoch schedule에는 너무 짧아 peak lr
# 근처에서 중단됐다.
patience = EARLY_STOPPING_PATIENCE
best_val_rmse = float("inf")
epochs_without_improvement = 0
history = []

def run_epoch(loader, training, hindcast_lambda):
    model.train(training)
    squared_error_sum = 0.0
    value_count = 0
    hindcast_squared_error_sum = 0.0
    hindcast_value_count = 0.0
    diag_sums = {}
    diag_batches = 0
    align_kl_sum = 0.0
    align_batches = 0
    surge_probs = []
    surge_labels = []
    for batch in loader:
        images = batch["images"].to(DEVICE, non_blocking=PIN_MEMORY)
        wind = batch["wind"].to(DEVICE, non_blocking=PIN_MEMORY)
        target = batch["target"].to(DEVICE, non_blocking=PIN_MEMORY)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(DEVICE.type, enabled=USE_AMP):
                prediction, aux = model(images, wind, return_aux=True)
                mse = F.mse_loss(prediction, target)
                loss = torch.sqrt(mse + RMSE_EPSILON)
                # v5a hindcast: 이미지 전용 전파 branch가 관측된 최근
                # 13개 wind(s=-72..0h = wind[:, 7:])를 재구성한다.
                # target은 관측값 그 자체 -- correspondence label이
                # 아니라 고정 시점 재구성이라 label 없이 성립한다.
                # 배정(어느 이미지가 어느 시점 기여)은 모델의
                # (tau_t, p_t)가 kernel로 암묵 수행. wind persistence
                # 지름길이 없는 경로의 supervision이라는 점이 v2/v3
                # prior 사멸의 해법이다. modality drop된 샘플
                # (image_keep=0)은 상수 재구성이라 제외.
                if aux["hindcast"] is not None:
                    hindcast_target = wind[:, 7:]
                    keep = aux["image_keep"].unsqueeze(-1)
                    kept_values = (
                        keep.sum() * hindcast_target.size(1)
                    ).clamp(min=1.0)
                    hindcast_mse = (
                        (aux["hindcast"] - hindcast_target) ** 2 * keep
                    ).sum() / kept_values
                    loss = loss + hindcast_lambda * torch.sqrt(
                        hindcast_mse + RMSE_EPSILON
                    )
                    # ballistic 잔차(tanh 출력 = delta/24h)의 L2:
                    # tau를 backbone 근방에 잡아둔다 (자유 tau의
                    # "값을 아무 시점에나 배치" 퇴화 방지).
                    loss = loss + TRANSIT_RESIDUAL_L2 * (
                        aux["transit_residual"] ** 2
                    ).mean()

                # v5b surge BCE: label은 미래 target에서 파생 (max_h W
                # - last_wind > threshold). 이미지 전용 head라 wind
                # 지름길이 없고, 출력 확률이 fusion gate로 들어가
                # "quiet면 닫고 pre-surge면 연다"의 판별 특징이 된다.
                # modality drop 샘플은 상수 logit이라 제외.
                if aux["surge_logit"] is not None:
                    surge_label = (
                        (target.max(dim=1).values - wind[:, -1])
                        > SURGE_THRESHOLD_KMS / 1000.0
                    ).float().unsqueeze(-1)
                    keep = aux["image_keep"].unsqueeze(-1)
                    surge_bce = F.binary_cross_entropy_with_logits(
                        aux["surge_logit"], surge_label,
                        weight=keep,
                        pos_weight=surge_pos_weight,
                        reduction="sum",
                    ) / keep.sum().clamp(min=1.0)
                    loss = loss + SURGE_LAMBDA * surge_bce

                # v7 backmapping alignment KL: label에서 역산한 소스
                # 궤적 분포로 kernel weight의 위치를 감독한다
                # (alignment_kl docstring / config ALIGN_* 참고).
                if aux["source_weight"] is not None:
                    align_kl, _ = alignment_kl(
                        model, aux, wind, target
                    )
                    loss = loss + ALIGN_LAMBDA * align_kl
                    align_kl_sum += float(align_kl.detach())
                    align_batches += 1
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        error_km_s = (prediction.detach() - target) * 1000.0
        squared_error_sum += float(torch.sum(error_km_s ** 2).cpu())
        value_count += error_km_s.numel()
        # mechanism 지표: hindcast RMSE (km/s). val RMSE가 노이즈에
        # 묻혀도 전파 branch가 학습되는지 이 값으로 판정한다 --
        # climatology(std_val ~= 90)보다 유의미하게 낮아져야 한다.
        if aux["hindcast"] is not None:
            keep = aux["image_keep"].unsqueeze(-1)
            hindcast_error_km_s = (
                aux["hindcast"].detach() - wind[:, 7:]
            ) * 1000.0
            hindcast_squared_error_sum += float(
                (hindcast_error_km_s ** 2 * keep).sum().cpu()
            )
            hindcast_value_count += float(
                (keep.sum() * hindcast_error_km_s.size(1)).cpu()
            )

        # 전파 branch 진단 (val 한정 -- train은 masking 증강이 섞여
        # 통계가 오염된다). 성공 signature: src_speed_std가 0에서
        # 벗어나고(상수 430 탈출), coverage/alpha가 오르고, arrival이
        # 레짐별로 분화. 실패 signature: 전부 초기값 부근 = 이 모델은
        # persistence + correction일 뿐이라는 뜻.
        if (not training) and (model.last_source_speed_kms is not None):
            batch_diag = {
                "src_speed_mean_kms": float(model.last_source_speed_kms.mean()),
                "src_speed_std_kms": float(model.last_source_speed_kms.std()),
                "arrival_mean_h": float(model.last_arrival_hours.mean()),
                "arrival_std_h": float(model.last_arrival_hours.std()),
                "source_gate_mean": float(model.last_source_gate.mean()),
                "coverage_hind_mean": float(model.last_coverage[:, :13].mean()),
                "coverage_future_mean": float(model.last_coverage[:, 13:].mean()),
                "alpha_mean": float(model.last_fusion_alpha.mean()),
            }
            for key, value in batch_diag.items():
                diag_sums[key] = diag_sums.get(key, 0.0) + value
            diag_batches += 1

        # surge AUROC 재료 (val 한정).
        if (not training) and (aux["surge_logit"] is not None):
            surge_probs.append(
                torch.sigmoid(aux["surge_logit"].detach())
                .squeeze(-1).cpu().numpy()
            )
            surge_labels.append(
                ((target.max(dim=1).values - wind[:, -1])
                 > SURGE_THRESHOLD_KMS / 1000.0)
                .cpu().numpy()
            )

    forecast_rmse = math.sqrt(squared_error_sum / value_count)
    hindcast_rmse = (
        math.sqrt(hindcast_squared_error_sum / hindcast_value_count)
        if hindcast_value_count > 0
        else float("nan")
    )
    diag = {
        key: value / diag_batches for key, value in diag_sums.items()
    }
    # v7 정렬 지표: KL(라벨 역산 궤적 || kernel weight). 하락 추세가
    # "위치 감독이 실제로 정렬을 움직인다"의 1차 신호. train/val 모두
    # 계산되지만 history에는 val 것만 실린다 (**val_diag).
    if align_batches > 0:
        diag["align_kl"] = align_kl_sum / align_batches

    # surge AUROC (Mann-Whitney rank). 확률이 연속이라 tie 보정은 생략.
    if surge_probs:
        scores = np.concatenate(surge_probs)
        labels = np.concatenate(surge_labels).astype(bool)
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos
        if n_pos > 0 and n_neg > 0:
            ranks = np.empty(len(scores))
            ranks[scores.argsort()] = np.arange(1, len(scores) + 1)
            diag["surge_auroc"] = float(
                (ranks[labels].sum() - n_pos * (n_pos + 1) / 2)
                / (n_pos * n_neg)
            )

    return forecast_rmse, hindcast_rmse, diag

if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        # hindcast 가중치 curriculum: 초반에 크게(이미지 시간 정렬부터
        # 굳힘) -> 선형 감쇠 후 고정. 스윕하지 않는 고정 기본값.
        decay_progress = min(
            1.0, (epoch - 1) / HINDCAST_LAMBDA_DECAY_EPOCHS
        )
        hindcast_lambda = (
            HINDCAST_LAMBDA_START
            + (HINDCAST_LAMBDA_END - HINDCAST_LAMBDA_START)
            * decay_progress
        )
        train_rmse, train_hindcast_rmse, _ = run_epoch(
            train_loader, training=True, hindcast_lambda=hindcast_lambda
        )
        with torch.no_grad():
            val_rmse, val_hindcast_rmse, val_diag = run_epoch(
                val_loader, training=False, hindcast_lambda=hindcast_lambda
            )

        # 이번 에폭이 실제로 쓴 lr을 기록한 뒤 다음 에폭 값으로 step.
        # (group 0 = base. 물리 group은 PHYSICAL_LR_MULT배로 동행.)
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed = time.perf_counter() - started

        # mechanism 지표. beta_h: persistence의 climatology 복귀율
        # (h 단조 증가가 기대 방향). dist_eff: 유효 전파 거리 [30,55]h.
        # 나머지 전파 진단은 val_diag (run_epoch, val 전체 평균).
        beta_mean = float(torch.sigmoid(model.reversion_logit).mean())
        dist_eff_h = float(
            30.0 + 25.0 * torch.sigmoid(model.dist_eff_raw.detach())
        )
        alpha_mean = val_diag.get("alpha_mean", float("nan"))

        history.append({
            "epoch": epoch, "train_rmse_km_s": train_rmse,
            "val_rmse_km_s": val_rmse,
            "train_hindcast_rmse_km_s": train_hindcast_rmse,
            "val_hindcast_rmse_km_s": val_hindcast_rmse,
            "hindcast_lambda": hindcast_lambda,
            "beta_mean": beta_mean, "dist_eff_h": dist_eff_h,
            **val_diag,
            "learning_rate": learning_rate, "seconds": elapsed,
        })
        print(
            f"epoch={epoch:03d} train_rmse={train_rmse:.3f} val_rmse={val_rmse:.3f}"
            f" hind={val_hindcast_rmse:.1f} alpha={alpha_mean:.3f} beta={beta_mean:.3f}"
            f" vstd={val_diag.get('src_speed_std_kms', float('nan')):.0f}"
            f" cov={val_diag.get('coverage_hind_mean', float('nan')):.2f}"
            f" kl={val_diag.get('align_kl', float('nan')):.3f}"
            f" auroc={val_diag.get('surge_auroc', float('nan')):.3f}"
            f" D={dist_eff_h:.1f}"
            f" lr={learning_rate:.2e} seconds={elapsed:.1f}"
        )
        
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
