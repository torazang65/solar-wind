# Baseline Spatial Transformer V2.2

## 1. Reason for V2.2

The reported V2.1 run had reached `70.803 km/s` validation RMSE at epoch 51 by
the epoch-65 log. At that point training RMSE was `58.372`, temporal-attention
entropy was still about `0.898`, and the mean attention age remained close to
60 hours. Masking and signed deltas helped the wind-only score by about 5.4
km/s, but the Transformer did not learn a sufficiently selective temporal
representation.

The primary V2.1 bottleneck was before the Transformer:

```text
128 x 4 x 4 CNN map -> flatten 2048 -> Linear -> one 96-dimensional token
```

All 16 spatial cells were compressed into one vector at every timestamp. V2.2
removes that projection and gives the Transformer every cell separately.

## 2. Representation size

| Property | V2.1 | V2.2 |
| --- | ---: | ---: |
| Spatial tokens per timestamp | 1 | 16 |
| Timestamps | 20 | 20 |
| Token dimension | 96 | 128 |
| Memory tokens | 20 | 320 |
| Values delivered to Transformer | 1,920 | 40,960 |
| Parameters | 602,640 | 669,680 |

V2.2 exposes 21.3 times more token values while increasing parameters by only
11.1 percent. The increase is in retained representation, not a large MLP.

## 3. Image front end

Preprocessing and the CNN remain controlled relative to V2.1:

1. resize grayscale 193/211 Angstrom images to 64 by 64;
2. normalize linearly with `uint8 / 255`;
3. zero everything outside the centered radius-0.49 solar disk;
4. calculate signed six-hour differences after masking;
5. multiply differences by four and clamp to `[-1, 1]`;
6. concatenate two intensity and two delta channels;
7. apply the official baseline's 5x5 stem and three multi-scale Inception3D
   blocks.

All CNN temporal kernels remain one. The CNN processes morphology inside each
frame and produces `(128, 4, 4)` for each of the 20 timestamps.

## 4. Factorized spatial-temporal Transformer

Each CNN cell is projected independently from 128 to 128 dimensions. Learned
row and column positions and a sinusoidal timestamp position are added without
flattening the grid:

```text
(B, 20, 128, 4, 4)
-> (B, 20, 4, 4, 128)
```

One factorized block then applies:

1. spatial self-attention over 16 cells independently for every timestamp;
2. temporal self-attention over 20 timestamps independently for every cell;
3. a shared position-wise feed-forward network.

A full 320-token self-attention layer would create `320^2 = 102,400` attention
scores per head and sample. The factorized block uses:

```text
20 * 16^2 + 16 * 20^2 = 11,520
```

This is 8.9 times smaller while retaining all 320 token vectors. Spatial and
temporal attention use separate parameters because the two axes have different
semantics.

## 5. Forecast and wind paths

The L1-wind path remains independent from the image timeline:

```text
20 winds -> MLP -> wind residual -> horizon-specific linear baseline
```

Twelve forecast queries represent +6 through +72 hours. They cross-attend to
the complete 320-token image memory, then fuse with the global wind feature.
The final image correction remains gated and bounded by the horizon-specific
training residual scale.

The speed-derived timing prior is disabled by default in V2.2. V2.1's learned
strength stayed close to its initial value and did not create selective
attention. `V22_TIMING_PRIOR_STRENGTH` remains an ablation option but defaults
to zero.

## 6. Regularization and training defaults

| Setting | Default |
| --- | ---: |
| Image size | 64 |
| Batch size | 128 |
| Epoch limit | 80 |
| Early-stopping patience | 15 |
| Transformer width / heads / layers | 128 / 4 / 1 |
| Feed-forward width | 256 |
| Dropout | 0.20 |
| Timestamp masking | 0.15 |
| Whole image-path dropout | 0.25 |
| Peak learning rate | 3e-5 |
| Warmup | 3 epochs |
| Schedule | cosine to 1e-6 |
| AdamW weight decay | 0.02 |
| EMA decay | 0.995 |
| Wind auxiliary loss | 0.25 |
| Image residual L2 | 0.002 |

Ordinary row shuffling, linear normalization, no north/south flip, and no
chain-balanced resampling are retained from V2.1 so the full CUDA result tests
the representation change rather than a different data distribution.

## 7. Server execution

```bash
cd ~/solar-wind-taeuk
git switch taeukjung
git pull --ff-only origin taeukjung

mkdir -p /home/jovyan/logs
LOG="/home/jovyan/logs/baseline_v2_2_$(date +%Y%m%d_%H%M%S).log"

nohup bash scripts_taeukjung/run_baseline_v2_2_server_cuda.sh train \
  > "$LOG" 2>&1 &

echo "PID=$! LOG=$LOG"
tail -f "$LOG"
```

The default run has approximately 76 training batches per epoch and writes to
`/home/jovyan/outputs/baseline_v2_2_taeukjung`, so V2.1 is not overwritten.

After training:

```bash
bash scripts_taeukjung/run_baseline_v2_2_server_cuda.sh infer
bash scripts_taeukjung/run_baseline_v2_2_server_cuda.sh diagnose
```

The diagnostic command writes:

- `preprocessing_delta.png`;
- `temporal_attention_mean.png`;
- `spatial_attention_by_horizon.png`;
- `spatiotemporal_attention_selected.png`;
- `token_spectrum.png`;
- temporal/spatial attention and per-horizon CSV files;
- representation and entropy statistics in `summary.json`.

## 8. Verification completed

- 64 px MPS forward/backward with finite gradients;
- 128 px MPS forward/backward with finite gradients;
- memory shape `(B, 320, 128)`;
- attention shape `(B, 12, 20, 16)` with sum error below `1.2e-7`;
- exact zero outside the disk and in the first delta frame;
- exact wind-only fallback;
- real-data training and EMA checkpoint reload;
- full validation inference;
- test submission generation with shape `(3868, 13)`;
- all five diagnostic PNGs and CSV/JSON artifacts generated and inspected.

The local smoke model used only 128 training rows, so its RMSE is not a model
quality estimate. V2.2 should be judged by the complete server run against the
currently observed V2.1 best of `70.803`, V6 `67.118`, and V7 `67.328`.
