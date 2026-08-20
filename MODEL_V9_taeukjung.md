# Solar Arrival TCN V9

## Objective

V9 tests whether the full spatiotemporal Transformer is the source of the
validation instability seen in V6-V8. It keeps the CEA CNN but replaces the
Transformer encoder with two small causal temporal convolution networks and an
explicit, inspectable image-arrival gate.

## Architecture

```text
20 x (193 A, 211 A) images
  -> disk mask and observer-aligned CEA reprojection
  -> per-frame multi-scale CNN
  -> 2 x 4 spatial grid flattened per frame
  -> image-only causal TCN
  -> per-image source score and transit-time estimate
                              -> learned arrival gate for each of 12 horizons
20 wind observations          -> horizon-specific image context
  -> wind-only causal TCN      -> gated image residual
  -> wind residual             -> final prediction
fixed linear wind baseline ----^
```

The image and wind observations never share a same-timestamp token. This avoids
asserting that wind observed near Earth and an EUV image at the same row index
represent the same solar event.

## Learned Arrival Gate

For image timestep `t`, the image TCN predicts a transit time `tau_t` constrained
to 48-120 hours and a source score. The implied arrival time relative to the
forecast origin is

```text
arrival_t = tau_t - image_age_t
```

For forecast horizon `h`, the gate combines image-query similarity, source
score, and a soft timing term:

```text
logit(h,t) = content(h,t) + source(t)
             - strength * (arrival_t - h)^2 / (2 * sigma^2)
gate(h,:) = softmax(logit(h,:))
```

Unlike V8, neither the input wind nor a fixed ballistic longitude selects the
image. The CNN representation predicts the timing, the prior strength can
shrink toward zero, and the resulting `(12, 20)` gate is exported for analysis.

## Regularization

- 10% image-timestep masking;
- 10% full-image-modality dropout;
- 15% feature dropout;
- bounded image residual using 2.5 times the baseline residual scale;
- small image-residual L2 weight of 0.002;
- ordinary row shuffling and no north/south flip;
- no EMA by default.

The model has 182,754 parameters with the default 2 by 4 CEA grid, versus about
203k for V6-V8. It contains no Transformer encoder or quadratic self-attention.

## Diagnostics

`diagnose_solar_arrival_v9.py` writes:

- mean arrival-gate heatmap and CSV;
- expected image age by horizon;
- transit/source statistics by image age;
- CNN representation singular-value spectrum and effective rank;
- per-sample full-model and wind-only errors;
- event/quiet RMSE and image-path gain.

Run it with:

```bash
bash scripts_taeukjung/run_solar_arrival_v9_server_cuda.sh diagnose
```

## Local Directional Result

A controlled 64 px run used the first 1,024 train rows, first 512 checkpoint
selection rows, 10 epochs, learning rate `2e-4`, and full 1,199-row validation
inference. It reached:

```text
overall validation RMSE : 68.861 km/s
chain-macro RMSE        : 67.095 km/s
wind-only RMSE          : 76.646 km/s
image-path gain         : 7.785 km/s
```

The comparison runs scored V6 `69.764` and V8 `69.197`, so V9 improved this
directional control by 0.903 and 0.336 km/s respectively. The learned gate also
follows the intended horizon-dependent timing order. The full 128 px CUDA run
is still required for model selection.

## Server Run

```bash
git switch taeukjung
git pull --ff-only origin taeukjung

mkdir -p /home/jovyan/logs
LOG="/home/jovyan/logs/solar_v9_$(date +%Y%m%d_%H%M%S).log"
nohup bash scripts_taeukjung/run_solar_arrival_v9_server_cuda.sh train \
  > "$LOG" 2>&1 &
echo "PID=$! LOG=$LOG"
tail -f "$LOG"
```

Inference and diagnostics use the same defaults as training:

```bash
bash scripts_taeukjung/run_solar_arrival_v9_server_cuda.sh infer
bash scripts_taeukjung/run_solar_arrival_v9_server_cuda.sh diagnose
```
