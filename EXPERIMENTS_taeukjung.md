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
