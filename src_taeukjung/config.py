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
DATA_ROOT = Path(os.getenv("DATA_ROOT", "public_dataset/competition_dataset_6h"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs/baseline_6h"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(os.getenv("CACHE_DIR", str(OUTPUT_DIR / "resized_cache")))

# ==========================================
# 3. Hyperparameters
# ==========================================
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "64"))
CHANNELS = ("193", "211")
RMSE_EPSILON = 1e-8
EPOCHS = int(os.getenv("EPOCHS", "20"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))

# ==========================================
# 4. Image Preprocessing
# ==========================================
SOLAR_DISK_MASK = os.getenv("SOLAR_DISK_MASK", "1").lower() not in {"0", "false", "no"}
SOLAR_DISK_CENTER_FRACTION = (0.5, 0.5)  # (y, x)
SOLAR_DISK_RADIUS_FRACTION = float(os.getenv("SOLAR_DISK_RADIUS_FRACTION", "0.49"))
SOLAR_CEA_RADIUS_FRACTION = float(os.getenv("SOLAR_CEA_RADIUS_FRACTION", "0.42"))
SPATIAL_FEATURE_SIZE = int(os.getenv("SPATIAL_FEATURE_SIZE", "4"))
IMAGE_NORM = os.getenv("IMAGE_NORM", "soft_cubic")
SOFT_CUBIC_STRENGTH = float(os.getenv("SOFT_CUBIC_STRENGTH", "0.25"))

TRANSFORMER_KWARGS = {
    "d_model": 128,
    "wind_dim": 32,
    "nhead": 8,
    "encoder_layers": 2,
    "ff_dim": 256,
    "dropout": 0.1,
}

TILE_TRANSFORMER_KWARGS = {
    "tile_grid_size": int(os.getenv("TILE_GRID_SIZE", "8")),
    "d_model": int(os.getenv("TILE_D_MODEL", "128")),
    "wind_dim": int(os.getenv("TILE_WIND_DIM", "32")),
    "nhead": int(os.getenv("TILE_NHEAD", "8")),
    "encoder_layers": int(os.getenv("TILE_ENCODER_LAYERS", "2")),
    "ff_dim": int(os.getenv("TILE_FF_DIM", "256")),
    "dropout": float(os.getenv("TILE_DROPOUT", "0.1")),
}
TILE_TRANSFORMER_LR = float(os.getenv("LEARNING_RATE", "3e-4"))

SOLAR_PROBABILISTIC_KWARGS = {
    "d_model": int(os.getenv("SOLAR_D_MODEL", "96")),
    "wind_dim": int(os.getenv("SOLAR_WIND_DIM", "24")),
    "nhead": int(os.getenv("SOLAR_NHEAD", "8")),
    "encoder_layers": int(os.getenv("SOLAR_ENCODER_LAYERS", "1")),
    "ff_dim": int(os.getenv("SOLAR_FF_DIM", "192")),
    "dropout": float(os.getenv("SOLAR_DROPOUT", "0.15")),
    "distribution_rank": int(os.getenv("DISTRIBUTION_RANK", "3")),
}
SOLAR_PROBABILISTIC_LR = float(os.getenv("LEARNING_RATE", "2e-4"))
PROBABILISTIC_NLL_WEIGHT = float(os.getenv("PROBABILISTIC_NLL_WEIGHT", "5.0"))

# ==========================================
# 5. Device Setup
# ==========================================
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

USE_AMP = DEVICE.type == "cuda"
AMP_DEVICE_TYPE = "cuda" if USE_AMP else "cpu"
PIN_MEMORY = DEVICE.type == "cuda"

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
elif DEVICE.type == "mps":
    print("INFO: Using Apple MPS")
else:
    print("WARNING: CUDA/MPS Unavailable")

if "BATCH_SIZE" in os.environ:
    BATCH_SIZE = int(os.environ["BATCH_SIZE"])
elif IMAGE_SIZE >= 512 and DEVICE.type == "mps":
    BATCH_SIZE = 1
elif IMAGE_SIZE >= 512:
    BATCH_SIZE = 8
elif DEVICE.type == "mps":
    BATCH_SIZE = 64
elif DEVICE.type == "cpu":
    BATCH_SIZE = 32
else:
    BATCH_SIZE = 256
