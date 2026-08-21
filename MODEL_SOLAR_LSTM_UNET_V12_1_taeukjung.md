# Solar Lag LSTM U-Net V12.1

V12.1 keeps the V12 AR(2), neural wind residual, LSTM, horizon attention,
soft lag prior, and bounded correction heads. It changes only the image token
encoder and adds conservative training defaults after the original V12
multi-lag run showed rapidly growing image correction and train-validation
separation.

## Image Encoder

1. Apply the same soft solar-disk mask to the 193 and 211 Angstrom images.
2. Add the same two signed running-difference channels.
3. Process each frame with a partial U-Net using channels
   `12,16,24,40,56` over resolutions `64,32,16,8,4`.
4. Fuse the 4 px bottleneck with 8 px and 16 px skip features.
5. Stop the decoder at 16 px and pool directly to the existing `2 x 8`
   source-token grid. There is no image reconstruction objective.

The production configuration has 943,356 trainable parameters, close to the
931,116-parameter V12 CNN model. GroupNorm avoids batch-statistic dependence
when server memory requires a smaller batch.

## Guarded Ablation

The retired CNN `exp1_multilag` is not rerun. Both V12.1 experiments use
64 px images, batch size 64, zero loader workers, peak learning rate `3e-5`,
correction cap 1.25, correction L2 weight 0.10, modality dropout 0.25, and
early-stop patience 6.

1. `unet_fixed96_guarded`: one 96-hour lag prior.
2. `unet_multilag_guarded`: 72, 84, 96, 108, and 120-hour lag experts.

Run both sequentially:

```bash
bash scripts_taeukjung/run_solar_lstm_unet_v12_1_ablation_server_cuda.sh train
```

Run one configuration:

```bash
V12_1_EXPERIMENTS="unet_fixed96_guarded" \
  bash scripts_taeukjung/run_solar_lstm_unet_v12_1_ablation_server_cuda.sh train
```

Outputs are written below
`/home/jovyan/outputs/solar_lag_lstm_unet_v12_1_seed777`. The final summary is
`solar_lag_lstm_unet_v12_1_ablation_summary.csv`.

Inference uses the same experiment selection:

```bash
V12_1_EXPERIMENTS="unet_fixed96_guarded" \
  bash scripts_taeukjung/run_solar_lstm_unet_v12_1_ablation_server_cuda.sh infer
```

## Local Verification

The default local command performs one synthetic forward/backward pass and
checks the 2 x 8 spatial attention shape and exact AR/wind fallback when image
modality is disabled:

```bash
conda activate ASAI
bash scripts_taeukjung/run_solar_lstm_unet_v12_1_local_mps.sh smoke
```

The optional `train` action uses only 32 training and 16 validation rows for
one epoch. Its score is not a performance estimate.
