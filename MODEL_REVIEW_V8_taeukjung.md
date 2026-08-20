# Solar-Wind Model Audit and V8 Design

## Scope

This review covers the full Taeukjung path from cached PNG bytes to the 12-step
forecast. It compares the implemented V1-V7 models with solar-wind forecasting
research and records the design of V8. Scores from different papers are not
directly interchangeable because their target cadence, date splits, lead times,
and available prior-rotation inputs differ.

## Task and strongest local evidence

The competition input contains 20 observations at 6-hour cadence: five days of
193 and 211 Angstrom images and five days of L1 wind speed. The target is the
next 12 wind values, from +6 to +72 hours.

The closest paper is Son et al. (2023), which uses the same two wavelengths,
five-day history, 6-hour cadence, and 72-hour target. Its model uses a 3D CNN,
three Inception blocks, an image LSTM, and a dense wind branch. It reports
horizon RMSE increasing from 37.4 km/s at +6 hours to 68.2 km/s at +72 hours.

- https://doi.org/10.3847/1538-4365/ace59a

Recorded complete-server results in this branch are:

| Model | Main change | Best validation RMSE |
| --- | --- | ---: |
| V1 | CEA CNN + temporal summary + Student-t auxiliary loss | **66.889** |
| V6 | CEA CNN + causal wind branch + axial attention | 67.118 |
| V7 | V6 with factorized longitude/time attention | 67.328 |
| V3 | RMSE-only CEA CNN Transformer | 67.737 |
| V5 | Hand-crafted coronal-hole cell statistics | 71.228 |
| Linear last-wind baseline | Closed-form horizon regressions | 75.903-76.094 |

V1 remains the best verified server checkpoint. V8 is an experiment until its
complete 128 px CUDA result is below 66.889.

## Code audit

### Image decoding

`dataset.py` decodes each RGB PNG with PIL `convert("L")`, resizes it with
bilinear interpolation, stores uint8 grayscale in a memmap, then applies the
selected normalization while loading a sample. The original RGB planes are
strongly correlated but not equal, so grayscale conversion is a lossy mapping
of the supplied color encoding. It is kept in V8 to avoid changing the data
contract and architecture in one experiment. RGB reconstruction should be a
separate ablation.

The server launchers use linear `/255` normalization. Earlier local experiments
found only small, inconsistent gains from cubic transforms. V8 therefore keeps
linear normalization.

### Geometry

The model masks the off-disk background and approximately reprojects the disk
onto a cylindrical equal-area grid. The PNG files have no FITS WCS metadata,
so this is not a full Stonyhurst reprojection. The configured CEA radius of
0.42 samples the reliable inner disk while the 0.49 mask rejects the outside
background. This central selection is empirical, not an exact estimate of the
observed solar radius.

The encoder supplies projected 193/211 intensity, channel-relative darkness,
normalized longitude, absolute latitude, and line-of-sight reliability `mu`.
Absolute latitude encodes the approximate north/south symmetry relevant to
near-ecliptic wind, while learned latitude positions can still represent
asymmetry.

### Spatial encoder

V6/V7 use the V3 learnable CNN:

1. a 5x5 stride-2 stem;
2. multi-scale depthwise 3x3 branches with dilation 1, 2, and 4;
3. average, maximum, and minimum downsampling paths;
4. adaptive pooling into a latitude/longitude grid.

This encoder preserves both bright active regions and dark coronal-hole
morphology. V5's fixed means, quantiles, darkness fractions, and channel ratio
performed substantially worse, showing that hard threshold summaries discarded
useful morphology or failed under split shift.

### Wind path

V5-V8 fit a fixed horizon-wise regression from the latest wind observation and
add a causal depthwise-TCN correction. A local audit tested regression on all
20 wind values and summary/trend features. The best full-history linear model
scored 75.986 km/s, versus 75.903 for the chain-weighted last-value model. The
extra history does not improve validation linearly.

The learned wind-only branch also stays near 76 km/s in complete runs. V8 keeps
the branch but reduces its auxiliary loss weight from 0.20 to 0.05 so that an
unproductive auxiliary objective cannot dominate the image correction.

### Memory and decoder

V6 applies self-attention jointly to each latitude band's 20x8 time/longitude
tokens. V7 factorizes longitude and temporal attention. Their 0.210 km/s score
difference provides no evidence that more specialized self-attention improves
generalization.

V1 instead averages spatial tokens for a 20-step temporal Transformer and gives
the decoder both those temporal summaries and the original spatial tokens. V8
restores this simpler topology with the corrected V6 image encoder. This avoids
mixing every spatial token before the forecast query and keeps each token's
physical location meaningful.

### Sampling and validation

Sliding windows recover 28 contiguous training chains with lengths from 14 to
961, and 11 validation chains of length 109. A chain boundary can be caused by
a missing observation; it does not necessarily define an independent solar
regime. Full inverse-length sampling gives a 14-row fragment the same epoch
weight as a 961-row segment and samples with replacement, reducing unique row
coverage.

V8 returns to ordinary row shuffling, matching the competition's row-level
micro RMSE. Chain IDs remain in diagnostics so chain-macro error is still
reported. V5-V7 behavior is unchanged because their launchers continue to set
chain balancing on.

## Findings from related work

### Position matters more than raw capacity

Brown et al. (2022) report only a 2-3 percent advantage for attention models
over CNN alternatives. Their best Swin model reaches 72.21 km/s at four days.
The study also uses north/south image flips and finds that equatorial, darker,
larger coronal holes drive higher predictions. It removes the image LSTM because
highly autocorrelated image sequences add multicollinearity with little context.

- https://doi.org/10.1029/2021SW002976

Collin et al. (2025) obtain 68.1 km/s with a small polynomial model based on
coronal-hole area and location plus 27-day wind persistence. Medium grids, 4x3
and 6x3, give the best timeline RMSE; finer grids add complexity without useful
timeline gain. Central cells within about +/-45 degrees latitude and +/-30
degrees longitude are especially informative. Distribution remapping improves
high-speed peak detection but worsens timeline RMSE from 68.1 to 75.1 km/s for
the 4x3 model, so V8 does not apply output-distribution expansion to an RMSE
submission.

- https://doi.org/10.1029/2024SW004125

SpeedNet uses Stonyhurst projection, off-limb removal, intensity
standardization, and persistent coronal-hole maps. Its attribution maps focus
on coronal holes near the central meridian, and the binary-map variant is more
stable than full EUV in several solar-cycle phases. It reports 71.4 km/s at a
three-day lead over the full cycle.

- https://doi.org/10.1029/2025EA004523

Dhuri et al. (2025) report 53-58 km/s with a multimodal encoder-decoder, but the
model uses solar-wind history from previous solar rotations. That feature is
not present in this competition input, so the score is not an architecture-only
comparison.

- https://doi.org/10.3847/1538-4365/adf436

### Missing information sets a ceiling

EUV images expose persistent coronal holes well but show CME-driven wind poorly.
Brown et al. explicitly report weak behavior around CMEs and solar maximum.
Son et al. make the same limitation for the exact 193/211 task. No Transformer
can reconstruct coronagraph, magnetic-field, ICME-list, solar-cycle phase, or
27-day recurrence information that the competition does not provide.

## V8 implementation

`SolarWindBallisticTransformerV8` makes six controlled changes from V6:

1. reduce the CEA token grid from 4x8 to 4x4;
2. restore V1's temporal-summary plus raw-spatial memory topology;
3. apply a north/south flip to 50 percent of training samples;
4. replace inverse-chain sampling with ordinary row shuffling;
5. apply EMA weights for validation and checkpointing;
6. add a speed-conditioned ballistic attention prior and bound the correction.

For horizon `h`, baseline speed `v`, observation age `a`, and synodic rotation
rate `omega`, V8 estimates

```text
transit_hours = 1 AU / v
source_delta  = h - transit_hours + a
expected_lon  = -omega * source_delta
```

The decoder receives a Gaussian log-attention prior around `expected_lon` and
the solar equator. The prior strength is a learned non-negative scalar. It is
initialized to 1.0 and can be disabled with
`SOLAR_V8_PHYSICS_PRIOR_STRENGTH=0` for an ablation.

The image correction is bounded with a horizon-specific scale estimated from
training baseline residuals. A small residual L2 term and EMA reduce the large
epoch-to-epoch validation swings observed in V6/V7.

## Local controlled comparison

The directional test used seed 777, 64 px images, the same first 1,024 training
rows and 512 validation rows, batch size 64, and 10 epochs. The saved models
were then evaluated on all 1,199 validation rows.

| Model | Full validation RMSE | Chain-macro RMSE |
| --- | ---: | ---: |
| V6 | 69.764 | 68.016 |
| V8 | **69.197** | 68.017 |

V8 improves full-validation micro RMSE by 0.567 km/s. Horizon results are:

| Horizon | V6 RMSE | V8 RMSE | V8 - V6 |
| ---: | ---: | ---: | ---: |
| 6 h | 33.746 | 31.341 | -2.405 |
| 12 h | 47.858 | 47.113 | -0.745 |
| 18 h | 57.766 | 57.226 | -0.540 |
| 24 h | 64.905 | 64.180 | -0.725 |
| 30 h | 70.401 | 69.251 | -1.151 |
| 36 h | 74.208 | 73.171 | -1.037 |
| 42 h | 76.409 | 75.478 | -0.930 |
| 48 h | 77.576 | 77.018 | -0.558 |
| 54 h | 78.437 | 78.141 | -0.296 |
| 60 h | 79.155 | 78.964 | -0.191 |
| 66 h | 79.639 | 79.584 | -0.055 |
| 72 h | 79.657 | 80.061 | +0.404 |

This is a reduced local experiment, not evidence that V8 beats V1 or V6 on the
complete 128 px server run. The +72-hour regression also shows that the
ballistic prior becomes less reliable when the source time lies farther from
the observed interval.

## Server decision rule

Run V8 once with the committed defaults. Keep the best EMA checkpoint and use
the complete validation RMSE for ranking.

- below 66.889: V8 becomes the best single model;
- 66.889-67.118: useful V1/V6 ensemble candidate;
- 67.118-67.7: retain as a diversity model only after measuring blend error;
- above 67.7: do not continue tuning the same architecture.

If an ablation is needed, change only one item per run in this order:

1. `SOLAR_V8_PHYSICS_PRIOR_STRENGTH=0`;
2. `SOLAR_V8_NORTH_SOUTH_FLIP_PROBABILITY=0`;
3. `SOLAR_V8_EMA_DECAY=0`;
4. `CHAIN_BALANCED_SAMPLING=1`.

Each ablation must use a separate `OUTPUT_DIR`; otherwise its checkpoint and
history will overwrite the default V8 run.
