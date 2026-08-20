# Baseline V2.3: Global AR + Neural Residual + Image Transformer

## Purpose

V2.3 replaces the learned wind-only baseline in V2.2 with a stable global
autoregressive forecast. The neural network predicts bounded corrections
instead of learning the complete solar-wind trajectory from scratch.

This is an ARIMA-family model with order `(2, 0, 0)`, followed by two neural
residual paths:

1. a wind-history residual network;
2. the V2.2 solar-image spatial-temporal Transformer.

No validation or test successor row is used. Every forecast is generated only
from the 20 wind values and 20 image timestamps contained in that sample.

## Why AR(2)

The train split was reconstructed into 28 independent continuous chains. A
global AR model was fitted to 10,419 unique one-step transitions. Candidate
wind-only validation results were:

| Forecast | Validation RMSE (km/s) |
| --- | ---: |
| Last wind plus per-horizon affine calibration | 76.094 |
| Direct 20-lag ridge regression | about 76.213 |
| Recursive differenced AR | at least 83.19 |
| Global recursive AR(2), ridge 30 | 75.462 |

In normalized units, the fitted recurrence is approximately:

```text
v[t] = 0.02549733 - 0.3432359 * v[t-2] + 1.2818651 * v[t-1]
```

The AR(2) chain-macro validation RMSE is 73.570 km/s. It is stronger and more
stable than the previous learned wind branch before image information is used.

## Architecture

The final forecast is additive:

```text
prediction = fixed_AR2_forecast
           + bounded_wind_neural_residual
           + gated_bounded_image_residual
```

### Fixed AR path

The AR coefficients are estimated once from the training split and stored in
the checkpoint. Forecasting recursively generates all 12 future wind values.

### Wind residual path

The wind network receives 40 features:

- the latest absolute wind level;
- 20 history values relative to the latest level;
- 19 one-step first differences.

It predicts a 12-step correction. Each horizon is bounded by 0.75 times the
training AR residual scale for that horizon. The final layer is initialized to
zero, so training starts exactly from the AR(2) forecast.

### Image path

The image branch is unchanged from V2.2:

- two EUV channels plus two signed six-hour difference channels;
- circular solar-disk masking;
- the official multi-scale Inception3D front end;
- a 4 by 4 feature grid at each of 20 timestamps;
- factorized spatial and temporal attention over 320 tokens;
- 12 forecast queries and bounded gated residual fusion.

This preserves considerably more image information than V2.1 while keeping
the AR and image timelines separate until forecast decoding.

## Default Training Configuration

- image size: 64 px
- batch size: 128
- parameters: 672,240
- optimizer: AdamW, learning rate `3e-5`, weight decay `0.02`
- scheduler: 3-epoch warmup followed by cosine decay
- EMA decay: `0.995`
- wind auxiliary loss weight: `0.25`
- image time masking: `0.15`
- image modality dropout: `0.25`
- image residual L2 weight: `0.002`

## Verification

The following checks were completed locally in the `ASAI` conda environment:

- Python compilation and shell syntax checks;
- NumPy and Torch AR recursion agreement within `1e-7`;
- finite forward and backward passes on MPS;
- zero neural wind correction at initialization;
- short real-data training run;
- strict checkpoint reload and full validation/test inference;
- `(3868, 13)` submission generation;
- temporal and spatial attention diagnostics.

The short smoke checkpoint produced 75.411 km/s on the full validation split,
but it used only 256 training samples and is only an end-to-end pipeline check.
The full CUDA run is required for a performance conclusion.

## Server Commands

```bash
bash scripts_taeukjung/run_baseline_v2_3_server_cuda.sh train
bash scripts_taeukjung/run_baseline_v2_3_server_cuda.sh infer
bash scripts_taeukjung/run_baseline_v2_3_server_cuda.sh diagnose
```

Default outputs are written to
`/home/jovyan/outputs/baseline_v2_3_taeukjung`.
