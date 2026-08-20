# Solar Hybrid V10

## Purpose

V10 directly extends Seokho's V5b model and combines the stabilization
techniques validated on the Taeuk branch. The image path is allowed to add
arrival-time and surge information without being able to overwrite a strong
wind baseline without bounds.

## Architecture

The input is 20 solar frames and 20 in-situ wind measurements at six-hour
intervals.

1. Apply a soft-edge solar-disk mask to the raw 193/211 channels.
2. Compute signed running differences on the GPU and amplify them by four.
3. Feed `(193, 211, delta193, delta211)` into Seokho's V5b Inception3D.
4. Preserve a `2x4` latitude-longitude grid when forming image tokens.
5. Keep 20 image and 20 wind tokens separate in the Transformer encoder.
6. Use 12 future queries in a one-layer Transformer decoder.
7. Retain the image-only source-speed, transit, hindcast, and surge heads.

The final forecast is:

```text
forecast = recursive_AR2
         + bounded_neural_wind_residual
         + bounded_ballistic_propagation_anomaly
         + bounded_transformer_correction
```

AR(2) is fitted globally on reconstructed unique temporal chains rather than
duplicated sliding windows. The global propagation scale is anchored at the
empirical 96-hour lag. Image features still adjust speed-dependent arrival
time and intensity.

Each neural term is bounded by the horizon-specific AR residual standard
deviation. A wrong image or attention signal therefore cannot replace the AR
forecast with an unbounded correction.

## Training

- Main 12-horizon RMSE
- Auxiliary AR plus neural-wind RMSE
- Image-only hindcast of the latest 72 hours
- Surge BCE for a future rise greater than 100 km/s
- Transit-residual and component-amplitude regularization
- EMA checkpoint, AdamW, three-epoch warmup and cosine decay
- Time masking, image modality dropout, and correction dropout

Validation records micro RMSE, chain-macro RMSE, hindcast RMSE, surge AUROC,
and the RMS of every output component.

## Defaults

- Input: `64x64`
- Transformer: `d_model=128`, two encoder layers, one decoder layer, 8 heads
- Batch size: 128 on CUDA
- Up to 80 epochs, early-stopping patience 15
- Linear normalization and radius 0.49 soft-edge disk mask

## Commands

```bash
git switch taeukjung
git pull --ff-only origin taeukjung

bash scripts_taeukjung/run_solar_hybrid_v10_server_cuda.sh train
bash scripts_taeukjung/run_solar_hybrid_v10_server_cuda.sh diagnose
bash scripts_taeukjung/run_solar_hybrid_v10_server_cuda.sh infer
```

Different seeds are written to different output directories by default.

```bash
SEED=1234 bash scripts_taeukjung/run_solar_hybrid_v10_server_cuda.sh train
```

For local MPS:

```bash
conda activate ASAI
bash scripts_taeukjung/run_solar_hybrid_v10_local_mps.sh train
```

Use `solar_hybrid_v10_component_gain.csv` from `diagnose` to check whether the
wind neural term beats AR, propagation helps surge and fast-wind subsets, the
free Transformer term does not damage quiet cases, and gains are distributed
across temporal chains. Treat differences below 1 km/s on one seed as noise;
compare at least two seeds before selecting a submission model.
