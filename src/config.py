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
OUTPUT_DIR = Path("outputs/baseline_6h")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 3. Hyperparameters
# ==========================================
IMAGE_SIZE = 64
CHANNELS = ("193", "211")
RMSE_EPSILON = 1e-8
BATCH_SIZE = 256
EPOCHS = 20
NUM_WORKERS = 4

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