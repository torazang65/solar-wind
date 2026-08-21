# Solar Source Map V11.2

V11.2 is a controlled follow-up to the Seokho-centered V11.1 source map. It
does not add a new wind model or a free Transformer correction. The experiment
isolates image resolution, longitudinal source-map resolution, augmentation
correctness, and one-step forecast consistency.

## Architecture

Each sample contains 20 six-hour image pairs and 20 wind observations. A soft
solar-disk mask removes off-disk pixels. The image path concatenates raw images
with signed temporal differences and applies the same Inception3D encoder used
by V11.1.

The final feature map is pooled to a configurable `2 x C` grid. Every time-cell
source predicts speed, evidence, transit-time residual, and a longitude offset.
Cell longitude centers and offset limits are derived from `C`, so `2 x 4` and
`2 x 8` use the same physical equations without hard-coded coordinates. Solar
rotation and ballistic transit convert each source into a distribution over 13
hindcast and 12 forecast times. The source forecast is fused with V11.1's
persistence/mean-reversion base.

Trainable parameter counts are 288,052 for `2 x 4` and 419,124 for `2 x 8`.

## Corrected Augmentation

V11.1 zeroed dropped feature tensors but its source heads could recreate
nonzero evidence from biases and coordinates. V11.2 carries the augmentation
mask into the physical source calculation:

- A dropped time step has exactly zero source weight for every cell and target
  time.
- A dropped image modality has zero source weight and zero fusion alpha.
- A modality-dropped prediction is therefore exactly the wind base.
- Backmapping supervision excludes time steps removed by time masking.

## Consecutive Consistency

Experiment 4 recovers one-step chains from overlapping image filenames. The
loader verifies that consecutive rows share 19 images, 19 wind values, and 11
target values. For valid pairs, the auxiliary term is

```text
0.05 * RMSE(prediction_t[1:], prediction_t+1[:-1])
```

Paired samples share augmentation decisions over their 19 common images. The
term constrains only the overlapping forecast; the V7 forecast, hindcast,
backmapping, transit, and surge objectives remain unchanged.

## Four-Way CUDA Ablation

The integrated launcher runs these experiments in order:

1. 64 px, `2 x 4`, batch 256, corrected masks
2. 128 px, `2 x 4`, batch 64, corrected masks
3. 128 px, `2 x 8`, batch 64, corrected masks
4. 128 px, `2 x 8`, batch 32, corrected masks and consistency weight 0.05

```bash
bash scripts_taeukjung/run_solar_source_map_v11_2_ablation_server_cuda.sh train
bash scripts_taeukjung/run_solar_source_map_v11_2_ablation_server_cuda.sh diagnose
bash scripts_taeukjung/run_solar_source_map_v11_2_ablation_server_cuda.sh infer
```

Outputs are isolated below
`/home/jovyan/outputs/solar_source_map_v11_2_ablation_seed777`. Training writes
`solar_source_map_v11_2_ablation_summary.csv` after all four runs. Inference
strictly checks the checkpoint version, preprocessing, image size, mask, and
source grid before loading weights.

Common defaults are 35 epochs, learning rate `3e-5`, three warmup epochs,
patience 15, seed 777, linear image normalization, and no chain-balanced
resampling.
