# Taeuk architecture experiments

## Baseline kernel structure

The shared baseline applies one `Conv3d` with kernel `(1, 5, 5)`, followed by
three Inception blocks. Every block has parallel `(1, 1, 1)`, `(1, 3, 3)`, and
`(1, 5, 5)` convolution paths plus a `(1, 3, 3)` max-pooling path. All temporal
kernel dimensions are one, so the convolutions process spatial structure in
each frame without mixing adjacent timestamps. The stem and each Inception
block are followed by spatial max pooling.

This is a generic multi-scale image encoder. Its main limitation for this task
is that repeated maximum pooling naturally retains bright active regions while
it can discard the area and intensity of dark coronal holes.

## Literature-derived constraints

The competition data and baseline match the setup in the 2023 paper
"Three-day Forecasting of Solar Wind Speed Using SDO/AIA Extreme-ultraviolet
Images by a Deep-learning Model". That work reports that 64 px and 128 px
inputs perform almost identically and argues that coronal-hole position and
area matter more than fine boundary detail:

- https://doi.org/10.3847/1538-4365/ace59a

An attention-based EUV forecasting study reports that attention models beat
the tested convolutional alternatives by roughly 2-3 percent. Its diagnostics
recover three useful physical relationships: wider, darker, and more
equatorial coronal holes are associated with faster wind, while relevant disk
longitude depends on the wind travel time:

- https://doi.org/10.1029/2021SW002976

The CEA model therefore keeps longitude and latitude information through the
forecast-query stage instead of applying a fixed radial center weight.

For probabilistic forecasting, TACTiS-2 models flexible joint distributions
with an attentional copula, while low-rank Gaussian copula forecasting models
high-dimensional time-varying dependence efficiently. The local implementation
uses the smaller low-rank idea with a Student-t tail rather than importing a
large external forecasting framework:

- https://arxiv.org/abs/2310.01327
- https://arxiv.org/abs/1910.03002

## Residual distribution diagnostics

Diagnostics were calculated on the local training split after fitting the same
horizon-wise last-wind linear baseline used by the Transformer models.

- Horizon residual standard deviation: 26.19 to 88.41 km/s.
- Adjacent-horizon residual correlation: 0.82 to 0.96.
- The first three covariance eigenvectors explain 95.0 percent of variance.
- Residual excess kurtosis ranges from 7.54 at +6 h to 0.82 at +72 h.
- Residual skew is positive at every horizon, from 1.85 to 1.09.

These measurements justify a rank-3 heavy-tailed joint output. The submission
continues to use the conditional mean because the competition metric is RMSE.

## Local MPS results

All numbers below use the provided validation split and seed 777.

| Experiment | Parameters | Local setting | Best validation RMSE |
| --- | ---: | --- | ---: |
| Horizon-wise linear baseline | 24 | closed form | 76.094 |
| CNN-free tile Transformer | 528,865 | batch 128, 10 epochs | 70.446 |
| Compact tile Transformer | 242,953 | batch 256, 15 epochs | 70.866 |
| CEA probabilistic Transformer | 199,806 | batch 128, 2 completed epochs | 72.818 |

The CEA run was intentionally stopped during epoch 3 because MPS takes about
178 seconds per epoch. The epoch-2 checkpoint was loaded by the independent
inference script and reproduced validation RMSE 72.818 exactly. It also
generated a `(3868, 13)` test submission and a matching uncertainty CSV.

The short CEA result is a runtime and learning-path validation, not a converged
model comparison. CUDA training with checkpoint selection is required before
ranking it against the tile model.

## Important optimizer finding

The original compact Transformer computed MSE after both predictions and
targets were divided by 1000. Its output stayed close to the linear baseline
because upstream gradients became too small. The new scripts compute exactly
the same squared error in km/s units and clip the global gradient norm to 1.0.
On a fixed synthetic image/wind batch, the CEA probabilistic model reduced RMSE
from 12.64 to 1.46 km/s, confirming that image, attention, and distribution
paths all train.

## V3 geometry and objective corrections

V3 was created after inspecting the exact 128 px preprocessing output on real
193 A and 211 A images. It is a separate implementation; V1 and V2 are retained
unchanged for comparison.

The configured `0.49` disk radius is close to the image edge and is useful for
removing the square black background. A radial profile over 64 training images,
however, places the median EUV limb-brightening peak near `0.393`. V1 and V2
used `0.49` for both masking and CEA sampling, so off-limb corona and near-edge
background were stretched into the projected surface. V3 keeps the mask radius
at `0.49` and uses an independently configurable CEA radius of `0.42`.

V2 defined darkness as `1 - intensity`. Near low-`mu` CEA boundaries, masked
and interpolated values near zero therefore became maximum-strength darkness.
On the inspected sample, mean low-`mu` darkness was 0.937 and 0.839 for the two
channels. V3 instead computes the positive channel-wise deficit below a
`mu`-weighted disk reference, normalizes it by that reference, and multiplies
it by `sqrt(mu)`. The same low-`mu` diagnostic falls to 0.009 with no values
above 0.8.

V1 and V2 sent only the spatial mean of every frame through their temporal
encoder. V3 reshapes the 4 by 8 CEA grid into four latitude-band sequences. Each
sequence jointly attends over `20 timestamps * 8 longitude cells`, allowing a
feature to interact with neighboring longitudes as the Sun rotates. Forecast
queries then attend to all 640 contextualized tokens.

Finally, V3 removes the Student-t distribution heads and auxiliary NLL. Its
only optimization objective is MSE in `(km/s)^2`; this has exactly the same
minimizer as the competition RMSE. The server launcher also changes the default
image mapping from soft cubic to linear. CUDA validation performance is pending
and must be recorded before V3 replaces the best V1 checkpoint.

Local implementation checks:

- 128 px MPS forward and backward completed with finite gradients;
- image token shape `(1, 20, 32, 72)`, encoded memory `(1, 640, 96)`, output
  `(1, 12)`;
- 199,033 model parameters;
- actual cached dataset batch forward and backward completed;
- checkpoint strict reload reproduced outputs exactly;
- the optional mask-radius API preserved the original V2 mask and forward path.

The first 128 px CUDA run reached its best validation RMSE of `67.737` at epoch
7 and stopped at epoch 15. Training RMSE continued from 57.207 to 50.368 while
validation degraded, confirming that the remaining limit is generalization, not
optimization. This is 0.848 km/s worse than the earlier 128 px V1 best of
66.889, so V3 does not replace V1 based on this run.

The original CUDA log displayed `residual_rms=inf` from epoch 2 onward. This was
only a diagnostic overflow: AMP returned the residual in FP16, and squaring a
correction above roughly 256 km/s exceeds the FP16 range. Prediction errors were
already cast to FP32, so the loss, reported RMSE, checkpoint, and inference were
valid. V3 now casts the residual to FP32 before accumulating its RMS diagnostic.

## V4 Cartesian projection control

The CEA mapping remains approximate because the PNG data do not contain FITS
WCS metadata. V4 tests this assumption without changing the rest of V3. It
keeps the original observed disk, masks pixels beyond radius `0.49`, and supplies
normalized horizontal position, absolute vertical position, and projected
`mu = sqrt(1 - r^2)` coordinate channels. Relative darkness is zero outside the
disk and uses the same `sqrt(mu)` reliability as V3.

The local run uses 64 px, batch size 128, linear normalization, 12 maximum
epochs, seed 777, and the same 199,033-parameter temporal and forecast model.
Its output is isolated under `dev/outputs/solar_cartesian_v4_local_64`. This is
a geometry ablation, not a replacement for the 128 px CUDA V3 run.

The run reached its best validation RMSE of `70.385` at epoch 4. It remained
worse than the 128 px CEA runs and provides no evidence for removing CEA.

## V5 causal wind and explicit coronal-hole features

V5 addresses the V3 generalization gap without increasing capacity. It has
179,725 parameters and separates the forecast into three additive terms:

1. a fixed horizon-wise linear AR baseline from the latest wind;
2. a learned causal-TCN wind residual;
3. a Transformer residual from wind tokens and explicit CEA coronal-hole cells.

The fixed image feature vector contains intensity means and quantiles, relative
darkness means and area fractions, the 193/211 log ratio, CEA coordinates, and
their first temporal differences. This avoids asking a CNN to rediscover the
same low-intensity region measurements from only 28 independent training
chains.

The chain audit recovered 28 training chains totaling 9,607 rows. Their lengths
range from 14 to 961, so ordinary row shuffling gives the longest chain about 69
times as much influence as the shortest. V5 uses inverse-length weighted
linear-baseline fitting and sampling, and records chain-macro validation RMSE
alongside the competition micro RMSE. The fixed validation split contains 11
equal chains of 109 rows.

On the complete validation split, the chain-weighted fixed baseline scores
`75.903` micro RMSE and `73.879` chain-macro RMSE, versus `76.094` for the
original row-weighted baseline.

Implementation checks completed locally:

- 128 px MPS forward and backward: finite `(2, 12)` output;
- explicit feature width: 36 per time and CEA cell;
- one-epoch real-data smoke training with chain-balanced sampling completed;
- checkpoint reload, full validation inference, and `(3868, 13)` submission
  generation completed;
- the smoke run used only 64 training and 32 validation rows and is therefore a
  pipeline test, not a performance estimate.

The complete 128 px CUDA run reached its best validation RMSE of `71.228` at
epoch 8 and stopped after epoch 16. Wind-only validation RMSE remained near 76,
while image-residual RMS varied from 58 to 81 km/s. The fixed coronal-hole
statistics therefore did not generalize as well as the learned V3 encoder.

## V6 CNN restoration

V6 changes only the V5 image representation: the fixed CEA cell statistics are
removed and the complete V3 CEA CNN is restored. The chain-weighted baseline,
chain-balanced sampler, causal wind branch, axial Transformer, objective, and
optimizer remain unchanged. This makes the V5-to-V6 CUDA comparison an isolated
test of learned morphology versus hand-designed statistics.

Local implementation checks:

- 203,197 parameters;
- finite 128 px MPS forward and backward with output shape `(2, 12)`;
- real-data training with two DataLoader workers completed;
- strict checkpoint loading, full validation inference, and a `(3868, 13)`
  submission completed from the smoke checkpoint.
