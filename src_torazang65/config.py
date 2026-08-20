import os
import random
from pathlib import Path
import numpy as np
import torch

# ==========================================
# 1. Seed Setup
# ==========================================
# SEED=1234 python train.py 처럼 환경변수로 오버라이드. v2/v3 비교에서
# 그룹별 delta ±5가 학습 궤적 노이즈로 판명됐으므로, 아키텍처 변경의
# 판정은 seed 2개 이상 평균으로 한다. OUTPUT_DIR에 seed가 들어가
# 멀티시드 런이 서로 덮어쓰지 않는다.
SEED = int(os.environ.get("SEED", "777"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==========================================
# 2. Paths & Directories
# ==========================================
DATA_ROOT = Path("public_dataset/competition_dataset_6h")
# train.py가 시작 시 기존 best_model.pth를 지우므로, 런이 바뀔 때마다
# 이름을 올려 이전 산출물을 보존한다.
OUTPUT_DIR = Path(f"outputs/transformer_v4_diffchan_torazang65_seed{SEED}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path("outputs/cache")

# ==========================================
# 3. Hyperparameters
# ==========================================
IMAGE_SIZE = 64
CHANNELS = ("193", "211")
RMSE_EPSILON = 1e-8
BATCH_SIZE = 256
EPOCHS = 100
NUM_WORKERS = 4

# ---- 학습률 스케줄 ----
# v2 런의 병목: lr=1e-4에서 val 바닥이 epoch 4에 왔고, 이후 train은
# 계속 내려가는데 val은 다시 69.8 밑으로 못 왔다. ReduceLROnPlateau의
# 첫 감축(epoch 10)은 이미 과적합 구간이라 늦었다. peak를 낮추고
# cosine으로 1에폭부터 계속 식혀서 바닥을 늦고 깊게 만드는 것이 목표.
# zero-init prior가 학습될 시간을 버는 효과도 겸한다. PEAK_LR가 주된
# 스윕 대상.
PEAK_LR = 3e-5
MIN_LR = 1e-6
WARMUP_EPOCHS = 3

# ==========================================
# 3.5 Model architecture
# ==========================================
# train.py와 inference.py가 같은 구조를 쓰도록 여기서 한 번만 정의한다.
# 값이 어긋나면 체크포인트 load_state_dict가 실패한다.
#
# 유효표본이 수백 개 수준(9,607 샘플이지만 6시간 stride로 19배 중복)이라
# 이전 설정(d_model=256, enc3/dec2, 3.74M)은 과적합했다. e21에 val 65.951로
# 바닥을 치고 상승 전환.
# wind_dim은 제거됨: 이미지/wind가 타임스텝별로 한 토큰에 concat되던
# 구조에서 각자 별도 토큰 스트림(20+20=40 토큰)으로 분리되면서
# 두 모달리티 모두 full d_model을 쓴다. 기존 체크포인트와는 호환 안 됨.
#
# v2 (2x4pool_dynprior): attention/ablation 분석 결과를 반영한 두 가지
# 구조 변경. 이전 체크포인트와 호환 안 됨 (image_projection 1024-dim,
# prior 모듈 추가).
#  1) GAP -> 2x4 (lat x lon) pooling: 이미지 기여의 실체가 "도래
#     스트림 감지"인데 GAP가 코로나홀의 중앙/가장자리 위치를 지우고
#     있었다 (fast&quiet에서 이미지 gain이 유의미한 음수 = false
#     positive 정황). 대칭 2x2는 center cell이 없어 central meridian
#     거리 표현이 약하므로, longitude 4열을 그대로 보존하고 latitude만
#     반으로 접는 비등방 풀링을 쓴다.
#  2) dynamic temporal prior: 디코더 cross-attention의 시점 선택이
#     샘플 불변 고정 템플릿이었다 (slow-fast COM gap -2h vs 이론 +34h).
#     이미지에서 읽은 solar-state로 통과시간 tau를 예측해 attention
#     logit에 가우시안 bias를 더한다. gate=0이면 v1으로 퇴화.
# v4 (diffchan): 입력에 running-difference 채널 추가 (193, 211, Δ193,
# Δ211). CNN의 시간 커널이 전부 1이라 CME의 on-disk 신호(coronal
# dimming/플레어 증광 = 프레임 간 픽셀 차분)를 모델이 스스로 만들 수
# 없어 dataset.py에서 공급한다. 판정: 같은 스케줄인 v3 대비
# surge/event 지표(surge gain, event RMSE, worst-15) + seed 2개 평균.
# 전체 RMSE는 노이즈(±1.5)에 묻히므로 판정 기준이 아니다.
MODEL_KWARGS = dict(
    d_model=128,
    nhead=8,
    num_encoder_layers=2,
    num_decoder_layers=1,
    dim_feedforward=256,
    dropout=0.1,
    # 학습 중 이미지 토큰을 타임스텝 단위로 마스킹하는 확률.
    # 0으로 두면 증강이 꺼진다. 주된 스윕 대상.
    time_mask_prob=0.15,
    # 학습 중 샘플 단위로 이미지 스트림 전체를 drop하는 확률.
    # wind-only 경로가 항상 자립하도록 강제해서 이미지 경로 암기를
    # 억제한다. 0이면 꺼짐. time_mask_prob와 함께 스윕 대상.
    modality_drop_prob=0.25,
    # 2 = 원본 EUV 채널만 (v3까지), 4 = + running-diff 채널 (v4).
    # dataset.py가 항상 4채널을 만들므로 2로 되돌리려면 dataset의
    # 차분 concat도 함께 빼야 한다.
    image_in_channels=4,
)

# ==========================================

# ==========================================
# 4. Device Setup
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
PIN_MEMORY = DEVICE.type == "cuda"

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
else:
    print("WARNING: CUDA Unavailable")