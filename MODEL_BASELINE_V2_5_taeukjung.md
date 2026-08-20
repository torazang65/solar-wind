# Baseline V2.5: Deep Fixed-Lag Transformer

## Goal

V2.5 increases the trainable neural contribution while retaining V2.4's
physically constrained 96-hour image-to-wind mapping. It is a separate version
so that the stronger neural path can be compared directly with V2.4.

```text
prediction = fixed_AR2
           + neural_wind_residual
           + fixed_lag_attention_image_scale_residual
```

## Changes From V2.4

| Property | V2.4 | V2.5 |
| --- | ---: | ---: |
| Transformer encoder layers | 1 | 2 |
| Feed-forward dimension | 256 | 320 |
| Parameters | 672,240 | 953,520 |
| Image scale head | Linear | LayerNorm + 2-layer MLP |
| Maximum image scale | 15% | 30% |
| Initial image gate | 18% | 40% |
| Learning rate | `3e-5` | `1e-4` |
| Image modality dropout | 20% | 10% |
| Image time masking | 5% | 2% |
| Image residual cap | 1.0 x AR scale | 1.5 x AR scale |
| Residual L2 weight | `0.002` | `0.0005` |
| EMA decay | `0.995` | `0.99` |

The wind and image output heads use small nonzero weight initialization. This
lets gradients reach the wind encoder, image CNN, and attention blocks from
the first optimization step. V2.4's zero initialization initially trained
only the final output heads.

## Constraints Retained

- circular solar-disk masking;
- fixed global AR(2) baseline;
- fixed 96-hour lag center;
- 12-hour lag sigma;
- hard 24-hour source-time window;
- local 12-hour temporal self-attention;
- weak central-disk spatial prior;
- horizon-specific absolute residual bounds.

The stronger neural path therefore cannot discard the physical source-time
mapping or produce an unlimited image correction.

## Verification

- 64 and 128 px finite forward/backward checks passed;
- image CNN and wind encoder receive nonzero gradients on the first step;
- five-epoch, 256-row MPS smoke training completed without divergence;
- strict EMA checkpoint reload completed;
- full validation and test inference completed;
- `(3868, 13)` test submission and attention diagnostics were generated;
- no collapsed image-memory features were detected.

The small smoke checkpoint scored 75.242 km/s over the complete validation
split, compared with 75.426 for the matching V2.4 pipeline check. These values
are not performance estimates because both models used only 256 training rows.

At the V2.5 smoke checkpoint, the mean image gate was 0.400 and the mean
absolute image scale was 0.506%. V2.4's corresponding four-epoch scale was
about 0.005%, confirming that the neural image path now has a materially
larger effect.

## Server Commands

Default 64 px run:

```bash
bash scripts_taeukjung/run_baseline_v2_5_server_cuda.sh train
```

Recommended 128 px run:

```bash
IMAGE_SIZE=128 \
BATCH_SIZE=32 \
NUM_WORKERS=2 \
OUTPUT_DIR=/home/jovyan/outputs/baseline_v2_5_128_taeukjung \
bash scripts_taeukjung/run_baseline_v2_5_server_cuda.sh train
```

Use the same image size and output directory for `infer` and `diagnose`.
