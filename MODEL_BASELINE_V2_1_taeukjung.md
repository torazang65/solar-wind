# Baseline Transformer V2.1

## 1. Objective

V2.1 restarts from the official competition baseline instead of extending the
increasingly specialized V5-V9 models. It keeps the baseline's proven
multi-scale image CNN and whole-history wind MLP, then adds only four changes:

1. remove the black area outside the solar disk;
2. expose signed six-hour image changes as explicit input channels;
3. replace the image LSTM with one small temporal Transformer and one forecast
   cross-attention block;
4. regularize image corrections around a strong wind-only linear forecast.

The design target is validation generalization, not minimum training error.

## 2. Why restart from the baseline

The official baseline processes every image with temporal kernel size one,
uses multi-scale 1x1, 3x3, and 5x5 Inception branches, reduces each timestamp to
a 128 by 4 by 4 feature map, and sends the resulting 20 vectors to an LSTM. Its
wind branch is a separate MLP over all 20 measured wind values.

The completed experiments showed that extra physical structure did not
consistently improve validation RMSE:

| Model | Complete CUDA validation RMSE (km/s) | Observation |
| --- | ---: | --- |
| V3 | 67.737 | CEA geometry overfit after the early epochs |
| V6 | 67.118 | Best of the geometry/CNN family |
| V7 | 67.328 | Factorized Transformer did not beat V6 |
| V8 | 68.966 | Fixed ballistic bias was too restrictive |
| V9, 64 px | 69.174 at epoch 2 | Train RMSE fell below 51 while validation rose above 73 |

The data windows overlap heavily, so the nominal row count substantially
overstates the number of independent solar-wind events. V2.1 therefore has
about 603k parameters, uses one encoder layer, and applies several independent
forms of regularization.

## 3. Input and preprocessing

Each sample contains 20 timestamps at six-hour spacing over the previous five
days:

- EUV images: `(20, 2, H, W)` for 193 and 211 Angstrom channels;
- L1 wind: `(20,)`;
- forecast target: `(12,)`, covering +6 through +72 hours.

The default image size is 64 by 64. The dataset cache remains uint8 and stores
ordinary resized grayscale images. All experiment-specific operations happen
after loading, so the same cache can be reused safely.

### Solar disk mask

A circular mask centered at `(0.5, 0.5)` with radius `0.49 * min(H, W)` is
multiplied into every image before any difference is calculated. Pixels outside
the disk are exactly zero and cannot become edge features in the delta input.

### Linear intensity

V2.1 deliberately returns to simple `uint8 / 255`. The soft-cubic mapping is
not used by default because the prior ablations did not establish a stable
validation gain and nonlinear contrast can amplify calibration differences
between observation periods.

### Signed temporal difference

For each wavelength independently:

```text
delta[0] = 0
delta[t] = clamp(4 * (masked[t] - masked[t-1]), -1, 1)
```

A sample of real 64 px data had intensity standard deviation about `0.197` and
raw delta standard deviation about `0.047`; a fixed gain of four puts both
signals on comparable scales without a learned normalization that could fit a
specific validation chain. The CNN therefore receives four channels per time:

```text
[193 intensity, 211 intensity, 193 signed delta, 211 signed delta]
```

## 4. Architecture

### Image path

The image stem and spatial encoder follow the official baseline:

```text
4-channel input
-> Conv3d(4, 32, kernel=(1,5,5))
-> spatial max-pool
-> 3 x [Inception3D(1x1, 3x3, 5x5, pool branches) + spatial max-pool]
-> 128 x 4 x 4 per timestamp
-> flatten to 2048
-> Linear(2048, 96) + LayerNorm
-> one 4-head temporal Transformer encoder
```

All CNN temporal kernels have size one. The CNN learns morphology within each
image, while the Transformer alone models relationships among the 20 image
times. This separation preserves the baseline's image processing behavior and
makes temporal attention inspectable.

### Wind path

Wind is not paired with the image token at the same row index. The L1 wind
observed at a timestamp left the Sun several days earlier, so same-index fusion
would encode a false causal alignment.

```text
20 wind values
-> Linear(20, 128) + SELU
-> Linear(128, 64) + SELU
-> 12-step wind residual
```

The residual is added to a fitted horizon-specific linear persistence
baseline. The final layer starts at zero, so optimization begins at the stable
linear baseline rather than at a random forecast.

### Forecast decoder

Twelve learned queries represent +6 through +72 hours. One compact decoder
block applies query self-attention and then cross-attention to the 20 image
tokens. The decoder output is fused with the global wind feature only after the
image timeline has been encoded.

A weak timing bias provides a prior, not a hard routing rule. With latest wind
speed `v` in the dataset's scaled units:

```text
transit_hours = clamp(41.555 / v, 48, 144)
expected_image_age[h] = clamp(transit_hours - horizon[h], 0, 114)
bias = -0.5 * softplus(alpha) * ((age - expected_age) / 36)^2
```

`softplus(alpha)` starts at only `0.10` and remains trainable. Content attention
can override the prior, unlike V8's more prescriptive ballistic routing and
V9's learned per-image arrival subsystem.

### Bounded image correction

The model predicts:

```text
prediction = linear wind baseline + wind residual + gated image residual
```

The image residual is bounded per horizon to 1.5 times the training baseline's
residual standard deviation. Its gate starts at `sigmoid(-1.5)`, about `0.18`.
This prevents early image noise from immediately replacing the stronger wind
forecast and limits the magnitude of later overfitting.

## 5. Generalization controls

The default server run uses:

| Setting | Value |
| --- | ---: |
| Image size | 64 |
| Parameters | 602,640 |
| Transformer layers / heads | 1 / 4 |
| Model / feed-forward width | 96 / 192 |
| Dropout | 0.20 |
| Image time masking | 0.15 |
| Whole image-path dropout | 0.25 |
| AdamW peak learning rate | 3e-5 |
| Weight decay | 0.02 |
| Warmup | 3 epochs |
| Schedule | cosine to 1e-6 |
| EMA decay | 0.995 |
| Early-stopping patience | 20 |
| Wind auxiliary loss weight | 0.25 |
| Image residual L2 weight | 0.002 |

Ordinary row shuffling is used. Chain-balanced sampling is disabled because it
changes the empirical competition distribution and did not improve the later
full runs. North/south flipping is also disabled because it is not guaranteed
to preserve every geoeffective asymmetry.

## 6. Server commands

```bash
git switch taeukjung
git pull --ff-only origin taeukjung

mkdir -p /home/jovyan/logs
LOG="/home/jovyan/logs/baseline_v2_1_$(date +%Y%m%d_%H%M%S).log"

nohup bash scripts_taeukjung/run_baseline_v2_1_server_cuda.sh train \
  > "$LOG" 2>&1 &

echo "PID=$! LOG=$LOG"
tail -f "$LOG"
```

Inference and diagnostics use the same preprocessing environment:

```bash
bash scripts_taeukjung/run_baseline_v2_1_server_cuda.sh infer
bash scripts_taeukjung/run_baseline_v2_1_server_cuda.sh diagnose
```

Outputs are isolated under
`/home/jovyan/outputs/baseline_v2_1_taeukjung`. Inference evaluates the full
validation split and writes `baseline_v2_1_submission.csv`; test labels are
never used for model selection.

Use 128 px only as a controlled follow-up after the 64 px result:

```bash
IMAGE_SIZE=128 BATCH_SIZE=64 \
OUTPUT_DIR=/home/jovyan/outputs/baseline_v2_1_128_taeukjung \
bash scripts_taeukjung/run_baseline_v2_1_server_cuda.sh train
```

The adaptive block averaging keeps the Transformer dimensions unchanged, so
64 and 128 px checkpoints use the same parameter count. They are still stored
in separate output directories because their learned CNN weights are not
interchangeable.

## 7. Verification

Local verification completed with the `ASAI` conda environment:

- finite 64 and 128 px MPS forward/backward passes;
- exact zero outside the disk and exact zero for the first delta frame;
- normalized forecast-to-image attention;
- exact wind-only fallback;
- real-data training, EMA checkpoint reload, full validation inference, and a
  `(3868, 13)` submission;
- preprocessing, attention, and representation-spectrum PNG diagnostics.

The small local run is a pipeline check, not a performance estimate. Promotion
should be based on the complete 64 px CUDA validation RMSE and per-horizon CSV.
