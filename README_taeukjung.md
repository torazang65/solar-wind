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
