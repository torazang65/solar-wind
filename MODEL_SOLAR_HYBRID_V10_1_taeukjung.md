# Solar Hybrid V10.1

## Reason for the revision

V10 validation decomposition showed that its components did not generalize
equally:

| Validation variant | RMSE (km/s) | Gain against AR(2) |
| --- | ---: | ---: |
| AR(2) only | 75.451 | 0.000 |
| Neural wind branch | 76.507 | -1.056 |
| Wind plus propagation | 68.784 | +6.668 |
| Wind plus Transformer | 70.953 | +4.499 |
| Full V10 | 68.078 | +7.374 |

The free Transformer correction helped the surge subset, but hurt the fast and
quiet subsets. V10.1 changes component selection instead of adding capacity.

## Architecture

The image preprocessing, masked four-channel Inception3D encoder, separate
image/wind Transformer tokens, fixed 96-hour propagation anchor, hindcast head,
and surge head are inherited from V10.

The final forecast is:

```text
forecast = recursive_AR2
         + bounded_ballistic_propagation_anomaly
         + selective_gate * bounded_transformer_correction
```

The V10 neural wind residual is disabled by default because it degraded the
held-out validation score. Its parameters are frozen and the exact AR(2)
forecast remains the wind anchor.

The correction gate combines two signals:

1. The image-only surge probability raises the gate when a future increase is
   supported by solar imagery.
2. A fast-wind/quiet suppression term lowers the gate when the latest L1 wind
   is already above 550 km/s but the surge head is not active.

The gate is bounded and retains a 0.15 floor. This prevents the Transformer
path from disappearing completely while reducing the validation regimes where
V10's free correction was empirically harmful.

## Training defaults

- 64 px masked linear-normalized input
- AR(2), ridge 30, fixed 96-hour propagation reference
- `d_model=128`, two encoder layers, one decoder layer, eight heads
- learning rate `5e-5`, two warmup epochs
- cosine decay reaches its floor at epoch 18
- maximum 40 epochs, early-stopping patience 8
- EMA 0.995 and correction dropout 0.25
- neural wind residual mix 0.0 and wind auxiliary loss 0.0

## Commands

```bash
bash scripts_taeukjung/run_solar_hybrid_v10_1_server_cuda.sh train
bash scripts_taeukjung/run_solar_hybrid_v10_1_server_cuda.sh diagnose
bash scripts_taeukjung/run_solar_hybrid_v10_1_server_cuda.sh infer
```

The diagnostic compares raw and gated Transformer corrections on all, slow,
mid, fast, surge, and quiet validation subsets. It also verifies that the
reported full prediction is reconstructed exactly from AR, propagation, and
the gated correction.
