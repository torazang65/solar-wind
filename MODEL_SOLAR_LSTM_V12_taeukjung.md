# Solar Lag LSTM V12

V12 returns to the competition CNN-LSTM pipeline while replacing its image
front end and direct final-state decoder. It incorporates the strongest
findings from the Taeuk and Seokho experiments without making the ballistic
source map the only prediction path.

## Architecture

1. Apply a soft solar-disk mask to both 193 and 211 Angstrom images.
2. Concatenate the two signed running-difference channels on the GPU.
3. Extract per-frame Inception features with BatchNorm.
4. Preserve longitude in the final CNN pool, producing a `2 x 8` grid at
   64 px instead of the competition model's `4 x 4` grid.
5. Add fixed latitude/longitude coordinates and project the 16 spatial cells
   into a 256-dimensional frame token.
6. Feed all 20 frame tokens to a one-layer LSTM with hidden size 192.
7. Use 12 horizon queries to attend to all 20 LSTM outputs. A soft mixture of
   72, 84, 96, 108, and 120 hour lag kernels biases attention but does not
   replace learned attention.
8. Build a stable wind forecast from a train-only global AR(2) fit plus a
   bounded neural residual that reads all 20 observed wind values and their
   differences.
9. Add a bounded, horizon-specific, gated image correction to the wind base.

The main objective is forecast RMSE. A 0.20 wind-base auxiliary loss keeps the
non-image path useful. A weak 0.01 lag-alignment KL uses target-derived transit
times during training only. It guides attention but is not used at inference.
The image correction also has a small L2 penalty.

V12 is not a Transformer. Its temporal sequence model is the competition
LSTM, while only the horizon readout uses query attention.

## Controlled CUDA Ablation

All three runs use the same 64 px images, `2 x 8` grid, LSTM, optimizer, seed,
and loss weights. Only the lag assumption changes:

1. `exp1_multilag`: five soft lag experts and weak target-derived alignment.
2. `exp2_fixed96`: a single 96 hour prior and weak alignment.
3. `exp3_learned_only`: no lag prior and no target-derived alignment.

Run all combinations with one command:

```bash
bash scripts_taeukjung/run_solar_lstm_v12_ablation_server_cuda.sh train
```

Run selected combinations by name:

```bash
V12_EXPERIMENTS="exp1_multilag exp3_learned_only" \
  bash scripts_taeukjung/run_solar_lstm_v12_ablation_server_cuda.sh train
```

The launcher writes isolated outputs below
`/home/jovyan/outputs/solar_lag_lstm_v12_ablation_seed777` and creates
`solar_lag_lstm_v12_ablation_summary.csv`. It skips missing histories when a
subset is requested.

Inference uses the same selection mechanism:

```bash
V12_EXPERIMENTS="exp1_multilag" \
  bash scripts_taeukjung/run_solar_lstm_v12_ablation_server_cuda.sh infer
```

## Minimal Local Check

The default local action does not load the competition dataset or train an
epoch. It runs one synthetic 64 px forward/backward pass and verifies exact
wind-base fallback when the image modality is dropped:

```bash
conda activate ASAI
bash scripts_taeukjung/run_solar_lstm_v12_local_mps.sh smoke
```

An optional one-epoch data smoke run is limited to 128 train and 64 validation
rows:

```bash
bash scripts_taeukjung/run_solar_lstm_v12_local_mps.sh train
```

## Interpretation

The primary comparison is `exp1_multilag` versus `exp3_learned_only`. A lower
validation RMSE and lag KL for experiment 1 would support Seokho's lag prior.
If experiment 3 wins, the fixed physical range is constraining the LSTM. The
fixed-96 run distinguishes a useful broad lag range from a useful single lag.

Always inspect `val_wind_base_rmse_km_s` and
`val_image_correction_rms_km_s`. The image model is useful only if the final
RMSE beats the independently trained wind base without an exploding image
correction or a saturated gate.
