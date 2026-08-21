# Solar Timing Transformer V13

## Motivation

V12.1 showed that changing the CNN to a Lite U-Net did not solve the timing
identification problem. The guarded multi-lag run reached `70.868 km/s` at
epoch 3, while fixed 96 hours reached only `72.045 km/s`. The fixed-lag model
also raised its validation correction gate to `0.809` and correction RMS to
`44.488 km/s`. A more expressive free correction therefore learned the
training set faster without finding a transferable image-to-arrival mapping.

V13 implements Seokho's central observation directly: an image source's
predicted speed and its arrival time must be the same latent variable, rather
than two independently learned outputs. The Transformer chooses among physical
source candidates; it cannot emit an arbitrary forecast residual.

## Architecture

1. The 20 pairs of 193/211 images are disk-masked and concatenated with signed
   temporal differences.
2. The V12.1 partial U-Net preserves local and multi-scale structure, then
   pools each frame to a longitude-preserving `2 x 8` source grid.
3. Every source cell predicts one speed `v` in `[250, 900] km/s` and one
   evidence value.
4. The same speed determines transit time:

   `transit = effective_distance / v`

5. Solar rotation and image age determine each source's arrival time:

   `arrival = -longitude / omega + transit - image_age`

6. A two-layer, four-head Transformer evaluates 13 hindcast queries
   (`-72..0 h`) and 12 forecast queries (`6..72 h`). Its cross-attention logits
   receive a non-vanishing Gaussian arrival-time bias.
7. Each query output is the attention-weighted convex average of source
   speeds. There is no query MLP that can create an unconstrained correction.
8. The final prediction is a bounded move from the train-only AR(2) anchor
   toward the source prediction. Its maximum blend is `0.5`, and the move is
   additionally capped by the horizon-specific AR residual scale.

## Mask And Leakage Guarantees

- Off-disk pixels are softly masked.
- Randomly masked image times are removed from both learned attention and the
  backmapping target.
- A query can attend only to images acquired at or before that query time.
- Hindcast supervision uses only `wind[:, 7:]`; target values are never model
  inputs.
- Modality-drop samples have an exact AR output and contribute no image
  correction.
- The AR(2) coefficients and residual scale are fitted on training chains only.

## Losses

The primary loss is future forecast RMSE. Image timing receives direct but
decaying supervision from 13-step hindcast RMSE: weight `0.50` at the start,
linearly reduced to `0.10` by epoch 8. The full experiment also applies a weak
`0.01` target-derived longitude backmapping KL. Correction L2, gate L1, and
source-speed temporal smoothness prevent the V12.1 correction-growth failure.

The paired ablation changes only the backmapping KL:

- `v13_full_backmapping`: alignment weight `0.01`
- `v13_no_backmapping`: alignment weight `0.0`

## Commands

Run the CUDA architecture smoke test:

```bash
bash scripts_taeukjung/run_solar_timing_transformer_v13_ablation_server_cuda.sh smoke
```

Run both controlled experiments:

```bash
mkdir -p /home/jovyan/logs
LOG="/home/jovyan/logs/solar_v13_$(date +%Y%m%d_%H%M%S).log"
nohup bash scripts_taeukjung/run_solar_timing_transformer_v13_ablation_server_cuda.sh train \
  > "$LOG" 2>&1 &
echo "PID=$! LOG=$LOG"
tail -f "$LOG"
```

Run only the physically supervised experiment:

```bash
V13_EXPERIMENTS=v13_full_backmapping \
bash scripts_taeukjung/run_solar_timing_transformer_v13_ablation_server_cuda.sh train
```

Inference uses the same experiment selector and rejects any preprocessing
mismatch:

```bash
V13_EXPERIMENTS=v13_full_backmapping \
bash scripts_taeukjung/run_solar_timing_transformer_v13_ablation_server_cuda.sh infer
```

## Decision Rule

Compare validation RMSE and chain-macro RMSE first. A credible timing gain
should also lower hindcast RMSE, keep correction RMS bounded, preserve a source
speed standard deviation above zero, and avoid a correction gate pinned near
its maximum. If `full_backmapping` loses to `no_backmapping`, the target-derived
longitude prior is too rigid; the speed-locked Transformer itself remains the
relevant comparison.
