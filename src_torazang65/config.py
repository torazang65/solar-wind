import os
import random
from pathlib import Path
import numpy as np
import torch

# ==========================================
# 1. Seed Setup
# ==========================================
SEED = 777
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==========================================
# 2. Paths & Directories
# ==========================================
DATA_ROOT = Path("public_dataset/competition_dataset_6h")
OUTPUT_DIR = Path("outputs/transformer_v1_64px_residual_torazang65")
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