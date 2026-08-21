# Solar Source Map V11

V11 starts from Seokho's `torazang65` V7 source-map idea at commit `2df99c8`
and combines it with the strongest controls from the Taeuk models. It is not a
free Transformer correction model. Its only learned forecast contribution is a
physically routed, bounded image residual on top of a train-only AR(2) anchor.

## Architecture

1. Reconstruct the training rows into 28 independent temporal chains and fit
   one ridge-regularized recursive AR(2) process. Validation and test targets
   are never used to fit the AR coefficients.
2. Apply a soft circular solar-disk mask to both EUV channels. Add signed
   six-hour differences with gain 4, producing `(193, 211, d193, d211)`.
3. Run the official multi-scale Inception3D CNN and pool each of 20 frames to a
   `2 latitude x 4 longitude` grid. The 160 cells remain distinct.
4. A cell-shared head consumes 128 CNN values plus latitude and longitude. It
   predicts source speed, source evidence, bounded transit residual, and a
   within-cell longitude offset.
5. Source arrival is fixed by solar rotation and ballistic propagation:

   `arrival = -longitude / omega + D_eff / source_speed + residual - frame_age`

   `D_eff` is fixed by the 96-hour, 430 km/s prior. The head cannot move a
   predicted speed to an arbitrary time without changing the physical terms.
6. Gaussian arrival kernels reconstruct 13 observed hindcast points and 12
   forecast points. A label-derived backmapping distribution supervises the
   source location with KL divergence.
7. A small gate blends the source forecast toward the AR(2) anchor. The gate is
   reduced for fast, quiet wind and multiplied by actual kernel coverage. The
   resulting residual is bounded by 1.25 times the training AR residual scale.

The final equation is:

`forecast = AR(2) + bounded(alpha * (source_forecast - AR(2)))`

There is no neural wind residual, Transformer decoder, or free additive
correction. This directly reflects the V5/V10 decompositions where propagation
generalized but unconstrained corrections overfit.

## Training Controls

- main normalized RMSE;
- image-only hindcast RMSE, weighted from 0.70 to 0.10 over eight epochs;
- V7 backmapping KL with 20-degree longitude sigma;
- bounded transit-residual L2 and propagation-energy L2;
- image-only surge BCE for selective gating;
- time masking, modality dropout, gradient clipping, AdamW, warmup/cosine, EMA;
- chain-aware manifests and chain-macro validation RMSE;
- strict architecture, model-kwargs, and preprocessing checkpoint metadata.

## Server Commands

```bash
bash scripts_taeukjung/run_solar_source_map_v11_server_cuda.sh train
bash scripts_taeukjung/run_solar_source_map_v11_server_cuda.sh diagnose
bash scripts_taeukjung/run_solar_source_map_v11_server_cuda.sh infer
```

The default is 64 px, batch size 128, 35 epochs, and seed-isolated output under
`/home/jovyan/outputs/solar_source_map_v11_taeukjung_seed777`.

For a controlled 128 px run after the 64 px result:

```bash
IMAGE_SIZE=128 BATCH_SIZE=64 \
OUTPUT_DIR=/home/jovyan/outputs/solar_source_map_v11_128_taeukjung_seed777 \
bash scripts_taeukjung/run_solar_source_map_v11_server_cuda.sh train
```

Do not compare a partial-row smoke run with a full validation result. Promote
V11 only if `full_v11` beats both `ar_only` and the current image-model result,
and if the gain survives chain-macro, quiet, fast, and surge diagnostics.
