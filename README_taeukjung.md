# Taeuk Jung workspace

This branch keeps Taeuk's implementation separate from the shared baseline.

## Layout

- `src/`: shared baseline
- `src_taeukjung/`: mask and soft-cubic preprocessing implementation
- `scripts_taeukjung/run_preprocess_64_20epoch.py`: fixed four-way, 64 px, 20-epoch ablation runner

## Server workflow

Run the personal branch directly while it is under development:

```bash
git fetch origin
git switch taeukjung
git pull --ff-only origin taeukjung
```

Run the focused training configuration:

```bash
IMAGE_SIZE=64 \
EPOCHS=20 \
BATCH_SIZE=256 \
NUM_WORKERS=4 \
SOLAR_DISK_MASK=1 \
IMAGE_NORM=soft_cubic \
SOFT_CUBIC_STRENGTH=0.25 \
OUTPUT_DIR=outputs/baseline_6h_taeukjung \
python src_taeukjung/train.py
```

Run the four-way ablation:

```bash
python scripts_taeukjung/run_preprocess_64_20epoch.py \
  --image-size 64 \
  --epochs 20 \
  --batch-size 256 \
  --num-workers 4 \
  --run-name ablation_4way_64_20
```

Merge into `main` only after the branch result has been reviewed.

## Compact Transformer

The compact Transformer uses the same mask and soft-cubic preprocessing as
`src_taeukjung`, then applies a temporal convolution, a two-layer encoder, and
one cross-attention readout for the 12 forecast horizons.

Local Apple MPS:

```bash
./scripts_taeukjung/run_transformer_local_mps.sh train
./scripts_taeukjung/run_transformer_local_mps.sh infer
```

Competition server CUDA:

```bash
./scripts_taeukjung/run_transformer_server_cuda.sh train
./scripts_taeukjung/run_transformer_server_cuda.sh infer
```

Train on the competition server:

```bash
DATA_ROOT=/home/jovyan/public_dataset/competition_dataset_6h \
OUTPUT_DIR=/home/jovyan/outputs/transformer_taeukjung \
IMAGE_SIZE=64 \
EPOCHS=20 \
BATCH_SIZE=256 \
NUM_WORKERS=4 \
SOLAR_DISK_MASK=1 \
IMAGE_NORM=soft_cubic \
SOFT_CUBIC_STRENGTH=0.25 \
python src_taeukjung/train_transformer.py
```

Run validation evaluation and test inference with the same preprocessing:

```bash
DATA_ROOT=/home/jovyan/public_dataset/competition_dataset_6h \
OUTPUT_DIR=/home/jovyan/outputs/transformer_taeukjung \
IMAGE_SIZE=64 \
SOLAR_DISK_MASK=1 \
IMAGE_NORM=soft_cubic \
SOFT_CUBIC_STRENGTH=0.25 \
python src_taeukjung/inference_transformer.py
```

## CNN-free tile Transformer

This variant removes both the spatial CNN and temporal Conv1D. For every image
timestamp and channel it calculates masked `mean`, `min`, `max`, and `std`
features on an 8 by 8 tile grid. A small MLP projects the flattened spatial
statistics into one token per timestamp, followed by a two-layer temporal
Transformer and 12 horizon queries.

The old Transformer and its output directory remain unchanged. The tile model
uses `best_tile_transformer.pth` under a separate output directory.

Local Apple MPS:

```bash
./scripts_taeukjung/run_tile_transformer_local_mps.sh train
./scripts_taeukjung/run_tile_transformer_local_mps.sh infer
```

Competition server CUDA:

```bash
git fetch origin
git switch taeukjung
git pull --ff-only origin taeukjung
bash scripts_taeukjung/run_tile_transformer_server_cuda.sh train
bash scripts_taeukjung/run_tile_transformer_server_cuda.sh infer
```

Server defaults are 64 px, batch size 256, 60 epochs with early stopping,
masking enabled, soft cubic strength 0.25, an 8 by 8 tile grid, and learning
rate `3e-4`. Any value can be overridden inline, for example:

```bash
EPOCHS=30 BATCH_SIZE=128 TILE_GRID_SIZE=8 \
  bash scripts_taeukjung/run_tile_transformer_server_cuda.sh train
```

## Solar geometry probabilistic Transformer

This is the primary server candidate. It combines:

- an observer-aligned approximate CEA reprojection from the centered disk PNG;
- explicit brightness, darkness, longitude, absolute latitude, and `mu`
  reliability channels;
- spatial-only multi-scale kernels with effective 1, 3, 5, and 9 px receptive
  fields, with no temporal convolution;
- average, maximum, and minimum downsampling so both bright active regions and
  dark coronal holes survive;
- 4 by 4 spatial tokens for every timestamp, retained until the 12 forecast
  queries attend to all 320 spatiotemporal tokens;
- a rank-3 multivariate Student-t forecast distribution. The competition CSV
  uses its conditional mean, while inference also saves the marginal standard
  deviation for every horizon.

The CEA transform is an observer-aligned approximation, not a full Carrington
reprojection. Exact heliographic reprojection would require the FITS WCS,
observation angle, and ephemeris metadata that are absent from the PNG inputs.

Local Apple MPS:

```bash
bash scripts_taeukjung/run_solar_probabilistic_local_mps.sh train
bash scripts_taeukjung/run_solar_probabilistic_local_mps.sh infer
```

Competition server CUDA:

```bash
bash scripts_taeukjung/run_solar_probabilistic_server_cuda.sh train
bash scripts_taeukjung/run_solar_probabilistic_server_cuda.sh infer
```

The server script defaults to 64 px, batch size 128, 40 maximum epochs with
early stopping, learning rate `2e-4`, and probabilistic NLL weight 5.0. It
reuses `/home/jovyan/outputs/cache_taeukjung/64px` and writes to the separate
`/home/jovyan/outputs/solar_probabilistic_taeukjung` directory.

The implementation only uses packages already required by the baseline:
PyTorch, NumPy, pandas, Pillow, and matplotlib.

## Solar geometry probabilistic Transformer v2

V2 keeps the v1 training and inference paths unchanged. It makes two focused
image-front-end changes:

- CEA brightness and darkness are no longer multiplied by `mu`; `mu` remains
  available as an independent coordinate channel.
- Each timestamp keeps a 4 by 8 latitude-longitude grid, producing 32 spatial
  tokens instead of v1's 16. Across 20 timestamps, the forecast queries attend
  to 640 spatial tokens.

The server launcher defaults to 128 px, batch size 64, 25 maximum epochs,
learning rate `1e-4`, and dropout 0.25. V2 writes its checkpoint and outputs to
`/home/jovyan/outputs/solar_probabilistic_v2_taeukjung`, so it cannot overwrite
v1 results.

Local Apple MPS:

```bash
bash scripts_taeukjung/run_solar_probabilistic_v2_local_mps.sh train
bash scripts_taeukjung/run_solar_probabilistic_v2_local_mps.sh infer
```

Competition server CUDA:

```bash
bash scripts_taeukjung/run_solar_probabilistic_v2_server_cuda.sh train
bash scripts_taeukjung/run_solar_probabilistic_v2_server_cuda.sh infer
```

The rectangular token grid can be overridden without editing code:

```bash
SOLAR_V2_SPATIAL_HEIGHT=4 SOLAR_V2_SPATIAL_WIDTH=8 \
  bash scripts_taeukjung/run_solar_probabilistic_v2_server_cuda.sh train
```

## Solar geometry Transformer v3

V3 is the RMSE-focused successor to the probabilistic models. V1 and V2 remain
available for reproducibility. The main changes are:

- the black-background mask radius (`0.49`) is independent of the approximate
  CEA surface radius (`0.42`);
- the default image mapping is linear instead of soft cubic;
- the darkness channel is a center-weighted relative intensity deficit, so
  masked or interpolated zeros cannot become maximum-strength dark features;
- each latitude band jointly attends over all 20 timestamps and 8 longitude
  cells, allowing longitudinal solar rotation to be modeled before forecast
  queries read the 640 encoded tokens;
- the Student-t distribution and auxiliary NLL are removed. Training minimizes
  MSE in `(km/s)^2`, which has exactly the same optimum as the competition RMSE;
- visual channel dropout, Transformer dropout, AdamW weight decay, gradient
  clipping, validation-RMSE scheduling, and early stopping remain active.

Competition server CUDA:

```bash
git switch taeukjung
git pull --ff-only origin taeukjung
bash scripts_taeukjung/run_solar_geometry_v3_server_cuda.sh train
```

Inference uses the best validation checkpoint and writes both the validation
metrics and submission CSV:

```bash
bash scripts_taeukjung/run_solar_geometry_v3_server_cuda.sh infer
```

The server defaults to 128 px, batch size 64, 30 maximum epochs, learning rate
`1e-4`, a 4 by 8 spatial grid, CEA radius `0.42`, and linear normalization. It
writes to `/home/jovyan/outputs/solar_geometry_v3_taeukjung`, separate from all
earlier versions. Long runs can be started with:

```bash
mkdir -p /home/jovyan/logs
LOG="/home/jovyan/logs/solar_v3_$(date +%Y%m%d_%H%M%S).log"
nohup bash scripts_taeukjung/run_solar_geometry_v3_server_cuda.sh train \
  > "$LOG" 2>&1 &
echo "PID=$! LOG=$LOG"
tail -f "$LOG"
```

Local Apple MPS:

```bash
bash scripts_taeukjung/run_solar_geometry_v3_local_mps.sh train
bash scripts_taeukjung/run_solar_geometry_v3_local_mps.sh infer
```

The geometry and memory settings can be overridden without editing code:

```bash
SOLAR_CEA_RADIUS_FRACTION=0.42 \
SOLAR_V3_SPATIAL_HEIGHT=4 SOLAR_V3_SPATIAL_WIDTH=8 \
  bash scripts_taeukjung/run_solar_geometry_v3_server_cuda.sh train
```
