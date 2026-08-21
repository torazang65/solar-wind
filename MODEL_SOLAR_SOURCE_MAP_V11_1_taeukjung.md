# Solar Source Map V11.1

V11.1 is a controlled return to Seokho's V7 source-map model at commit
`2df99c8`. V11 combined too many Taeuk-side hypotheses, so this version removes
them and tests Seokho's discovery with only the requested solar-disk mask.

## Removed From V11

- train-only AR(2) anchor;
- fixed 96-hour transit prior;
- AR-residual propagation cap;
- fast-wind/quiet suppression;
- extra shared source MLP;
- EMA checkpoint averaging;
- component-energy regularization and gradient clipping.

## Retained From Seokho V7

- official BatchNorm Inception3D CNN;
- raw `193`, `211` channels plus signed six-hour differences at gain 1;
- `20 frames x 2 latitude x 4 longitude` cell source map;
- cell-shared linear speed, source-gate, transit-residual, and longitude heads;
- synodic solar rotation and ballistic arrival equation;
- learnable bounded effective distance and `+-24 h` transit residual;
- image-only hindcast reconstruction and label-derived backmapping KL;
- image-only surge classifier and sample/horizon convex fusion gate;
- correction-free output: `base + alpha * (source_forecast - base)`;
- V7 defaults: 64 px, batch 256, 35 epochs, peak LR `3e-5`, physical scalar
  LR multiplier 100, time mask 0.15, modality dropout 0.25, patience 15.

The base is Seokho's learnable persistence/mean-reversion curve:

`base_h = last_wind + beta_h * (climatology - last_wind)`

The source arrival time is:

`arrival = -longitude / omega + D_eff / source_speed + residual - frame_age`

## Taeuk Addition

The only model-level addition is a soft circular mask applied before signed
differences and the CNN. It prevents the off-disk black background from
becoming a positional shortcut. MPS-compatible pooling, chain-macro reporting,
strict checkpoint metadata, and portable launchers affect execution and
diagnostics rather than the forecast equation.

## Commands

```bash
bash scripts_taeukjung/run_solar_source_map_v11_1_server_cuda.sh train
bash scripts_taeukjung/run_solar_source_map_v11_1_server_cuda.sh diagnose
bash scripts_taeukjung/run_solar_source_map_v11_1_server_cuda.sh infer
```

Default outputs are isolated under
`/home/jovyan/outputs/solar_source_map_v11_1_taeukjung_seed777`.

Compare V11.1 directly with Seokho V7 and V11 using best validation RMSE,
chain-macro RMSE, hindcast RMSE, alignment KL, surge AUROC, source-map gain over
the Seokho base, and the epoch at which validation reaches its minimum.
