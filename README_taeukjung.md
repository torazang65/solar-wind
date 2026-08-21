# Taeuk Jung workspace

This branch keeps Taeuk's implementation separate from the shared baseline.

## Solar Hybrid V10

V10 starts from Seokho's V5b Inception3D, separated image/wind Transformer,
ballistic hindcast, and surge head. It adds a temporal-chain AR(2) anchor,
soft disk masking, amplified signed differences, a fixed 96-hour propagation
scale, and horizon-wise bounded residuals. See
`MODEL_SOLAR_HYBRID_V10_taeukjung.md` for the architecture and commands.

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

## Local Cartesian v4 ablation

V4 is a local control experiment for the approximate CEA assumption. It keeps
the V3 CNN, relative-darkness channel, 4 by 8 token grid, longitude-time axial
attention, residual baseline, and RMSE-aligned loss. The only geometry change
is that it retains the observed Cartesian solar disk instead of reprojecting it.

The local launcher defaults to 64 px, batch size 64, 12 epochs, and a separate
`dev/outputs/solar_cartesian_v4_local_64` output directory:

```bash
BATCH_SIZE=128 bash scripts_taeukjung/run_solar_cartesian_v4_local_mps.sh
```

This experiment should be compared against a 64 px CEA V3 run with the same
seed and training settings before attributing a difference to the projection.

## Solar physics Transformer v5

V5 keeps the 128 px observer-aligned CEA geometry but replaces the learned CNN
front end with explicit coronal-hole statistics. For every one of 20 timestamps
and every cell in a 4 by 8 latitude-longitude grid, it extracts:

- mean 193 A and 211 A intensity;
- 10th, 25th, and 50th intensity quantiles for both channels;
- relative-darkness mean and area fractions above 0.15 and 0.30;
- the mean log 193/211 channel ratio and CEA coordinates;
- first temporal differences of every feature.

A four-block causal TCN encodes wind, predicts a wind-only correction to the
fixed linear AR baseline, and supplies wind tokens to the longitude-time
Transformer. Forecast queries produce a second residual from the fused wind and
image memory. A small auxiliary wind-only loss keeps the two responsibilities
separated.

Rows are reconstructed into sliding-window chains from their image filenames.
The fixed linear baseline and training sampler both use inverse-chain-length
weights, and validation reports both the competition micro RMSE and a
chain-macro RMSE. The recovered split contains 28 training chains with lengths
from 14 to 961 and 11 validation chains of length 109. A chain manifest and
best-epoch validation predictions are saved with the checkpoint.

Competition server CUDA:

```bash
git switch taeukjung
git pull --ff-only origin taeukjung
bash scripts_taeukjung/run_solar_physics_v5_server_cuda.sh train
bash scripts_taeukjung/run_solar_physics_v5_server_cuda.sh infer
```

Local Apple MPS:

```bash
bash scripts_taeukjung/run_solar_physics_v5_local_mps.sh train
bash scripts_taeukjung/run_solar_physics_v5_local_mps.sh infer
```

Both launchers default to 128 px, linear normalization, CEA radius `0.42`, a
4 by 8 grid, learning rate `1e-4`, dropout 0.25, and chain-balanced sampling.
The server uses batch size 64 and the local launcher uses batch size 4.

## CNN restoration v6

V6 is an isolated response to the V5 validation result. It preserves V5's
chain-weighted linear baseline, chain-balanced sampler, causal wind encoder,
Transformer, and RMSE-aligned loss, but replaces the fixed quantile/area image
statistics with the complete V3 CEA CNN encoder. This restores multi-scale
dilated kernels, relative-darkness channels, dual-polarity downsampling, and
learned 4 by 8 spatial tokens.

Competition server CUDA:

```bash
git switch taeukjung
git pull --ff-only origin taeukjung
bash scripts_taeukjung/run_solar_cnn_v6_server_cuda.sh train
bash scripts_taeukjung/run_solar_cnn_v6_server_cuda.sh infer
```

V6 writes to `/home/jovyan/outputs/solar_cnn_v6_taeukjung`, so it cannot
overwrite V3 or V5 checkpoints. Transformer findings and the proposed ablation
order are recorded in `TRANSFORMER_REVIEW_taeukjung.md`.

## Factorized Transformer v7

V7 keeps every V6 component fixed except the encoder attention layout. Instead
of flattening 20 timestamps and 8 longitudes into a length-160 sequence for
each latitude, it applies attention over 8 longitude cells and then over 20
timestamps. Both axes share the same multi-head-attention parameters, so V7 has
only 192 more parameters than V6 while reducing encoder attention scores from
102,400 to 17,920 per sample.

Competition server CUDA:

```bash
git switch taeukjung
git pull --ff-only origin taeukjung
bash scripts_taeukjung/run_solar_factorized_v7_server_cuda.sh train
bash scripts_taeukjung/run_solar_factorized_v7_server_cuda.sh infer
```

V7 defaults to 128 px and writes to the isolated
`/home/jovyan/outputs/solar_factorized_v7_taeukjung` directory.

## Ballistic Transformer v8

V8 keeps the V6 CEA CNN but uses a smaller 4 by 4 grid and the simpler V1
memory topology. Forecast-query attention receives a speed-conditioned prior
that follows solar rotation and an estimated Sun-to-L1 transit time. Training
uses north/south flips, ordinary row shuffle, a bounded residual, residual L2,
and EMA checkpoint weights. See `MODEL_REVIEW_V8_taeukjung.md` for the complete
audit and literature comparison.

Server training and inference:

```bash
bash scripts_taeukjung/run_solar_ballistic_v8_server_cuda.sh train
bash scripts_taeukjung/run_solar_ballistic_v8_server_cuda.sh infer
```

V8 defaults to 128 px and writes to
`/home/jovyan/outputs/solar_ballistic_v8_taeukjung`.

## Learned-arrival TCN v9

V9 removes the full Transformer encoder. A CEA CNN and causal TCN encode the
image sequence, a second causal TCN independently encodes wind, and a learned
per-image transit/source gate builds one image context for each forecast
horizon. This prevents same-index image/wind fusion and makes temporal routing
directly inspectable.

Server training, inference, and diagnostics:

```bash
bash scripts_taeukjung/run_solar_arrival_v9_server_cuda.sh train
bash scripts_taeukjung/run_solar_arrival_v9_server_cuda.sh infer
bash scripts_taeukjung/run_solar_arrival_v9_server_cuda.sh diagnose
```

The server defaults to 128 px, batch size 64, learning rate `2e-4`, a 2 by 4
CEA grid, soft-cubic strength 0.25, and an isolated output directory at
`/home/jovyan/outputs/solar_arrival_v9_taeukjung`. The local launcher requires
the `ASAI` conda environment. Architecture and diagnostics are documented in
`MODEL_V9_taeukjung.md`.

## Baseline Transformer v2.1

V2.1 restarts from the official Inception3D baseline after V9's 64 px run
overfit after epoch 2. It masks the solar disk, adds signed six-hour image
differences, replaces the image LSTM with one small temporal Transformer and
forecast cross-attention block, and applies a bounded gated image correction
to a strong wind-only forecast. Image and L1-wind timelines remain separate
until the forecast decoder.

Server training, inference, and diagnostics:

```bash
bash scripts_taeukjung/run_baseline_v2_1_server_cuda.sh train
bash scripts_taeukjung/run_baseline_v2_1_server_cuda.sh infer
bash scripts_taeukjung/run_baseline_v2_1_server_cuda.sh diagnose
```

The default is 64 px with 602,640 parameters, warmup plus cosine decay, EMA,
time masking, image-path dropout, and ordinary row shuffling. Outputs go to
`/home/jovyan/outputs/baseline_v2_1_taeukjung`. See
`MODEL_BASELINE_V2_1_taeukjung.md` for the full architecture and rationale.

## Baseline Spatial Transformer v2.2

V2.2 removes V2.1's `2048 -> 96` per-timestamp bottleneck. The official
Inception3D front end still produces a 4 by 4 map, but all 16 cells at all 20
timestamps are retained as 320 independent 128-dimensional tokens. A
factorized block applies spatial attention and temporal attention separately,
then the 12 forecast queries read the complete token memory. The default
ballistic timing prior is disabled.

```bash
bash scripts_taeukjung/run_baseline_v2_2_server_cuda.sh train
bash scripts_taeukjung/run_baseline_v2_2_server_cuda.sh infer
bash scripts_taeukjung/run_baseline_v2_2_server_cuda.sh diagnose
```

The default server run uses 64 px, batch size 128, and an isolated output at
`/home/jovyan/outputs/baseline_v2_2_taeukjung`. See
`MODEL_BASELINE_V2_2_taeukjung.md` for dimensions, diagnostics, and commands.

The completed V2.2 CUDA run reached its best validation RMSE of `69.653` at
epoch 26. Retaining 320 image tokens improved V2.1, but it remained behind V6.

## AR-Neural Baseline Transformer v2.3

V2.3 fits one global ARIMA-family `(2, 0, 0)` forecast on the reconstructed
training chains, then learns bounded wind-history and V2.2 image-Transformer
residuals. The fixed AR path alone reaches `75.462` validation RMSE, compared
with `76.094` for the previous affine last-wind baseline.

```bash
bash scripts_taeukjung/run_baseline_v2_3_server_cuda.sh train
bash scripts_taeukjung/run_baseline_v2_3_server_cuda.sh infer
bash scripts_taeukjung/run_baseline_v2_3_server_cuda.sh diagnose
```

The default is 64 px with 672,240 parameters. Outputs go to
`/home/jovyan/outputs/baseline_v2_3_taeukjung`. See
`MODEL_BASELINE_V2_3_taeukjung.md` for the architecture and leakage controls.

## Fixed-Lag Attentive Magnitude Transformer v2.4

V2.4 addresses the missing image-to-wind time labels by fixing the empirical
four-day solar-wind lag. Forecast cross-attention is centered on the mapped
source image and hard-limited to a 24-hour neighborhood. Spatial and local
temporal attention remain active inside that physically plausible region.

The image path predicts only a bounded percentage adjustment to the V2.3
AR-neural wind forecast, rather than generating an independent forecast curve.

```bash
bash scripts_taeukjung/run_baseline_v2_4_server_cuda.sh train
bash scripts_taeukjung/run_baseline_v2_4_server_cuda.sh infer
bash scripts_taeukjung/run_baseline_v2_4_server_cuda.sh diagnose
```

The default is 64 px with 672,240 parameters and writes to
`/home/jovyan/outputs/baseline_v2_4_taeukjung`. See
`MODEL_BASELINE_V2_4_taeukjung.md` for the fixed mapping and 128 px command.

## Deep Fixed-Lag Transformer v2.5

V2.5 retains V2.4's 96-hour lag and hard source-time window but gives the
neural paths more capacity and influence. It uses two Transformer layers, a
two-layer image-scale MLP, a 30% scale limit, a 40% initial image gate, small
nonzero output-head initialization, and a higher learning rate.

```bash
bash scripts_taeukjung/run_baseline_v2_5_server_cuda.sh train
bash scripts_taeukjung/run_baseline_v2_5_server_cuda.sh infer
bash scripts_taeukjung/run_baseline_v2_5_server_cuda.sh diagnose
```

The default 64 px model has 953,520 parameters. See
`MODEL_BASELINE_V2_5_taeukjung.md` for the controlled V2.4 comparison and the
recommended 128 px command.

## Selective Solar Hybrid V10.1

V10.1 keeps V10's masked Seokho V5b image encoder, fixed-lag propagation path,
and compact Transformer, but applies the validation decomposition directly.
The neural wind residual is disabled, AR(2) remains the exact wind anchor, and
the free Transformer correction is gated by image-only surge probability and
fast-wind/quiet suppression.

```bash
bash scripts_taeukjung/run_solar_hybrid_v10_1_server_cuda.sh train
bash scripts_taeukjung/run_solar_hybrid_v10_1_server_cuda.sh diagnose
bash scripts_taeukjung/run_solar_hybrid_v10_1_server_cuda.sh infer
```

The default is 64 px and writes seed-isolated outputs under
`/home/jovyan/outputs/solar_hybrid_v10_1_taeukjung_seed777`. See
`MODEL_SOLAR_HYBRID_V10_1_taeukjung.md` for the component evidence and gate.

## Cell Source Map V11

V11 ports Seokho V7's `20 x 2 x 4` cell source map and backmapping alignment
onto the Taeuk train-only AR(2) anchor. The masked Inception3D path predicts
cell speed, evidence, longitude offset, and bounded transit residual; solar
rotation and ballistic propagation determine when each cell can affect Earth.
The only learned forecast term is a horizon-bounded source residual. There is
no free Transformer correction or neural wind residual.

```bash
bash scripts_taeukjung/run_solar_source_map_v11_server_cuda.sh train
bash scripts_taeukjung/run_solar_source_map_v11_server_cuda.sh diagnose
bash scripts_taeukjung/run_solar_source_map_v11_server_cuda.sh infer
```

The default is 64 px, batch size 128, 35 epochs, EMA, and seed-isolated output
under `/home/jovyan/outputs/solar_source_map_v11_taeukjung_seed777`. See
`MODEL_SOLAR_SOURCE_MAP_V11_taeukjung.md` for equations, controls, and the
controlled 128 px command.

## Seokho-Centered Source Map V11.1

V11.1 removes V11's AR(2), fixed 96-hour lag, residual cap, fast/quiet
suppression, source MLP, and EMA. It restores Seokho V7's learnable effective
distance, persistence/mean-reversion base, cell-shared linear heads, and
unbounded convex source fusion. The only model-level Taeuk addition is the soft
solar-disk mask.

```bash
bash scripts_taeukjung/run_solar_source_map_v11_1_server_cuda.sh train
bash scripts_taeukjung/run_solar_source_map_v11_1_server_cuda.sh diagnose
bash scripts_taeukjung/run_solar_source_map_v11_1_server_cuda.sh infer
```

The launcher matches the Seokho V7 defaults: 64 px, batch size 256, 35 epochs,
peak learning rate `3e-5`, and physical-parameter multiplier 100. See
`MODEL_SOLAR_SOURCE_MAP_V11_1_taeukjung.md` for the controlled-difference list.

## Source Map V11.2 Three-Way Ablation

V11.2 keeps the V11.1 model center while correcting time/modality masking,
making the source grid dynamic, and optionally enforcing one-step chain
forecast consistency. One launcher compares 64 px `2 x 4`, 128 px `2 x 8`,
and 128 px `2 x 8` with consistency. The retired 128 px `2 x 4` experiment is
no longer run or included in the summary.

```bash
bash scripts_taeukjung/run_solar_source_map_v11_2_ablation_server_cuda.sh train
bash scripts_taeukjung/run_solar_source_map_v11_2_ablation_server_cuda.sh diagnose
bash scripts_taeukjung/run_solar_source_map_v11_2_ablation_server_cuda.sh infer
```

See `MODEL_SOLAR_SOURCE_MAP_V11_2_taeukjung.md` for the controlled variables,
mask guarantees, chain-pair definition, and output paths.

## Solar Lag LSTM V12

V12 keeps the competition LSTM sequence model and replaces its image front end
with a soft disk mask, signed temporal differences, and a longitude-preserving
`2 x 8` spatial grid. Twelve horizon queries read the full LSTM sequence using
an optional speed-dependent soft lag prior. Forecasts are anchored by a
train-only global AR(2) fit plus a neural encoder over all 20 observed wind
values; images contribute only a bounded gated correction.

Run the controlled multi-lag, fixed-96-hour, and learned-only comparison:

```bash
bash scripts_taeukjung/run_solar_lstm_v12_ablation_server_cuda.sh train
```

Run the minimal local architecture check:

```bash
bash scripts_taeukjung/run_solar_lstm_v12_local_mps.sh smoke
```

See `MODEL_SOLAR_LSTM_V12_taeukjung.md` for architecture details, selected-run
commands, metrics, and interpretation rules.

## Guarded Lite U-Net LSTM V12.1

V12.1 retains V12's train-only AR(2) wind anchor, complete LSTM sequence,
horizon attention, and lag prior while replacing only the image CNN with a
partial U-Net token encoder. The decoder fuses 4, 8, and 16 px features and
stops before full-resolution reconstruction, then pools to the same `2 x 8`
grid. Its 943,356 parameters remain close to the V12 control.

The retired V12 CNN exp1 is not rerun. One command compares guarded fixed-96
and multi-lag U-Net configurations:

```bash
bash scripts_taeukjung/run_solar_lstm_unet_v12_1_ablation_server_cuda.sh train
```

See `MODEL_SOLAR_LSTM_UNET_V12_1_taeukjung.md` for selected-run commands,
regularization settings, output paths, and inference.

## Speed-Locked U-Net Timing Transformer V13

V13 removes V12.1's free neural wind residual and free image-correction head.
The Lite U-Net predicts a speed for every time/latitude/longitude source cell;
that exact speed also determines its physical transit time. A Transformer uses
causal, physics-biased cross-attention to select source cells for 13 hindcast
and 12 forecast queries. Predictions remain anchored to train-only AR(2), with
at most a 0.5 bounded move toward the image-derived source speed.

One command compares weak target-derived backmapping against no backmapping:

```bash
bash scripts_taeukjung/run_solar_timing_transformer_v13_ablation_server_cuda.sh train
```

See `MODEL_SOLAR_TIMING_TRANSFORMER_V13_taeukjung.md` for the equations,
mask guarantees, diagnostics, selected-run command, and interpretation rules.

## Physics-Guided Deformable Timing V14

V14 retains V13's speed-locked source values and AR guard but replaces dense
source lookup with eight physically initialized samples per query and head.
Seven samples may move within 12 hours and 1.5 longitude cells; one physical
anchor remains fixed, and all samples are causally clipped. See
`MODEL_SOLAR_DEFORMABLE_TIMING_V14_taeukjung.md` for the ablation command.

## Direct Peak-Time and Peak-Value V15

V15 adds explicit future-event supervision to V14. One branch predicts which
of the 12 future bins contains the maximum, and another predicts that maximum
speed. A small capped event curve connects both outputs to the forecast while
preserving exact AR fallback under image masking. See
`MODEL_SOLAR_PEAK_EVENT_V15_taeukjung.md` for the two-run command.

## Native Longitude Profile LSTM V16

V16 removes learned spatial downsampling entirely. It retains all 64 original
longitude columns, extracts disk-masked mean/min/max/std and signed temporal
differences, and feeds a compact stride-one profile encoder and one-layer
LSTM. Native, scrambled-image, and wind-only controls run from one
server command documented in `MODEL_SOLAR_NATIVE_PROFILE_LSTM_V16_taeukjung.md`.
