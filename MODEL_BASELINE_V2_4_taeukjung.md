# Baseline V2.4: Fixed-Lag Attentive Magnitude Correction

## Hypothesis

Earlier models allowed temporal attention to discover the Sun-to-Earth source
time without direct source-time labels. Their attention often learned image
trends but did not establish a stable mapping between each image and forecast
horizon.

V2.4 supplies that missing mapping as a physical prior. Empirical coronal-hole
forecast models commonly use a four-day delay for solar-wind speed, and
WindNet activation analysis found relevant coronal-hole features roughly
three to four days before the predicted wind.

References:

- https://doi.org/10.1029/2024SW004125
- https://doi.org/10.1002/2016SW001390
- https://doi.org/10.1029/2020SW002478

The four-day delay is an empirical background-wind prior, not a claim that
every transient or CME has the same travel time.

## Time Mapping

The data contain 20 images at six-hour cadence, from 114 hours before the last
observation through the current time. The targets cover +6 through +72 hours.
For forecast horizon `h`, V2.4 fixes the source-image age to:

```text
source_age(h) = 96 hours - h
```

This maps the 12 targets to source ages `90, 84, ..., 24` hours and therefore
to `image_04, image_05, ..., image_15`.

## Constrained Attention

Attention remains part of the model, but it cannot freely choose an unrelated
time:

1. Spatial self-attention selects relevant cells within each image.
2. Temporal self-attention is local, with a default radius of 12 hours.
3. Forecast cross-attention receives a fixed Gaussian time bias centered on
   the four-day source mapping.
4. Tokens more than 24 hours from the mapped source time are hard-masked.
5. A weak central-disk spatial prior favors near-equatorial, near-central
   cells while learned attention can still select other visible regions.

The default Gaussian sigma is 12 hours. The time center, sigma, and hard
window are configuration values stored in the checkpoint, not learned
parameters.

## Forecast Structure

The wind path is inherited from V2.3:

```text
wind_prediction = fixed_global_AR2 + bounded_wind_neural_residual
```

The image path does not create another unconstrained 12-step wind curve.
Instead, it predicts a bounded percentage adjustment to each value in
`wind_prediction`:

```text
scale_fraction = 0.15 * tanh(image_scale_logit)
image_residual = bounded(wind_prediction * scale_fraction) * image_gate
prediction = wind_prediction + image_residual
```

The percentage limit is 15% by default. The absolute correction is also
bounded by the horizon-specific AR residual scale. The scale head is initialized
to zero, so the model starts exactly from the AR(2) forecast.

## Image Representation

- 193A and 211A intensity channels;
- signed six-hour differences for both channels;
- circular solar-disk masking;
- official multi-scale Inception3D front end;
- 4 by 4 spatial grid for every timestamp;
- 320 tokens of width 128;
- spatial, local-temporal, and constrained cross-attention.

## Default Configuration

- image size: 64 px
- parameters: 672,240
- fixed lag: 96 hours
- lag sigma: 12 hours
- lag hard window: 24 hours
- local temporal radius: 12 hours
- image scale limit: 15%
- learning rate: `3e-5`
- EMA decay: `0.995`
- time masking: `0.05`
- image modality dropout: `0.20`

## Verification

The following local checks completed successfully:

- finite 64 px forward and backward passes;
- exact zero wind and image residuals at initialization;
- actual mean attention ages within 0.10 hours of all 12 fixed source ages;
- four-epoch, 256-row real-data MPS smoke training;
- strict EMA checkpoint reload;
- full validation and test inference;
- `(3868, 13)` submission generation;
- attention, representation, preprocessing, and image-scale diagnostics.

The smoke checkpoint scored 75.426 km/s on the complete validation set. This
is only a pipeline check because the model saw 256 training rows.

A simple audit of central-disk mean brightness had almost no correlation with
the AR residual. This supports retaining nonlinear CNN and attention features
instead of replacing the image path with a single darkness statistic.

## Commands

Default 64 px server run:

```bash
bash scripts_taeukjung/run_baseline_v2_4_server_cuda.sh train
bash scripts_taeukjung/run_baseline_v2_4_server_cuda.sh infer
bash scripts_taeukjung/run_baseline_v2_4_server_cuda.sh diagnose
```

For a separate 128 px run:

```bash
IMAGE_SIZE=128 \
BATCH_SIZE=32 \
OUTPUT_DIR=/home/jovyan/outputs/baseline_v2_4_128_taeukjung \
bash scripts_taeukjung/run_baseline_v2_4_server_cuda.sh train
```

Inference and diagnostics must use the same three environment variables.
