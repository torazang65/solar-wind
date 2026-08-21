# Taeuk architecture experiments

## V10: Seokho V5b anchored hybrid

- Base: `src_torazang65/model.py` V5b
- Wind anchor: global AR(2) fitted on reconstructed temporal chains
- Image input: masked 193/211 plus signed running differences with gain 4
- Timing: global 96-hour anchor with a learned speed-dependent residual
- Output: bounded wind, propagation, and Transformer residuals
- Validation: micro RMSE, chain-macro RMSE, and component ablations
- Entrypoint: `scripts_taeukjung/run_solar_hybrid_v10_server_cuda.sh`

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

## V7 factorized attention

V7 changes only V6's Transformer encoder. One shared MHA kernel first attends
over 8 longitude cells independently for every timestamp and latitude, then
attends over 20 timestamps independently for every latitude-longitude cell. A
single feed-forward block follows both attention passes.

| Property | V6 | V7 |
| --- | ---: | ---: |
| Parameters | 203,197 | 203,389 |
| Encoder attention scores per sample | 102,400 | 17,920 |
| 128 px MPS forward/backward | finite | finite |

The parameter increase is only 192, while attention-score storage and work are
reduced by a factor of 5.7. Real-data smoke training with two DataLoader workers,
strict checkpoint reload, full validation inference, and `(3868, 13)` test
submission generation all completed. The smoke checkpoint used only 32 training
rows and is not a performance estimate.

The complete 128 px CUDA run reached its best validation RMSE of `67.328` and
chain-macro RMSE of `65.953` at epoch 7. V6 remains 0.210 km/s better in micro
RMSE and 0.283 km/s better in chain-macro RMSE, so factorized attention did not
replace V6.

## V8 ballistic decoder and regularized sampling

V8 returns to V1's simpler temporal-summary plus raw-spatial memory topology
while keeping V6's corrected CEA CNN. It uses a 4 by 4 grid, north/south flip
augmentation, ordinary row shuffling, EMA checkpoint weights, a bounded image
correction, and a speed-conditioned time/longitude attention prior derived from
solar rotation and ballistic transit time.

In a controlled local directional test with 64 px images, 1,024 training rows,
512 model-selection rows, and 10 epochs, full validation inference scored
`69.197`, versus `69.764` for V6 trained under the corresponding control. V8
improved 11 of 12 horizons; +72 hours was 0.404 km/s worse. This is a local
screening result only. The complete 128 px CUDA run must beat the verified V1
score of `66.889` before V8 is promoted.

The full code and literature audit is in `MODEL_REVIEW_V8_taeukjung.md`.

The complete 128 px CUDA run reached its best validation RMSE of `68.966` at
epoch 8. This is worse than V6 `67.118` and V1 `66.889`; the fixed ballistic
attention bias is therefore not promoted.

## V9 separate TCNs and learned image arrival

V9 removes Transformer self-attention and separates the image and wind time
paths. The CEA CNN produces a compact 2 by 4 representation for each image, an
image-only causal TCN estimates source strength and 48-120 hour transit time,
and a learned `(forecast horizon, image timestep)` gate aggregates the image
sequence. A separate wind TCN predicts the wind-only residual before gated
image fusion.

Implementation verification:

- 182,754 parameters with the default configuration;
- finite 64 px MPS forward/backward and normalized arrival weights;
- exact wind-only fallback when images are disabled;
- real-data smoke training and strict checkpoint reload completed;
- full validation inference and `(3868, 13)` submission generation completed;
- diagnostic CSV/JSON and four PNG plots generated successfully.

The controlled 64 px, 1,024-train-row, 512-selection-row, 10-epoch run at
learning rate `2e-4` scored `68.861` on the complete validation set and `67.095`
chain-macro RMSE. This improves the matching V6 control by 0.903 km/s and V8 by
0.336 km/s. The image path improved its wind-only prediction by 7.785 km/s, and
the gate followed the intended arrival-time ordering. A complete 128 px CUDA
run remains necessary.

## Baseline Transformer V2.1 restart

The complete V9 64 px log showed immediate generalization loss: validation RMSE
was best at `69.174` on epoch 2, while training RMSE continued from `64.151` to
below `51` and validation rose above `73`. V2.1 therefore restarts from the
official baseline CNN instead of extending V9's learned-arrival subsystem.

Changes relative to the official baseline:

- circular disk masking before feature extraction;
- two signed, scaled six-hour delta channels added to the two intensity channels;
- the official multi-scale Inception3D image CNN retained;
- image LSTM replaced by one 96-wide temporal Transformer and one forecast
  cross-attention block;
- image and L1-wind histories kept separate until forecast decoding;
- weak speed-derived timing bias, bounded gated image residual, time masking,
  modality dropout, warmup/cosine learning rate, and EMA.

Local implementation checks completed on MPS at both 64 and 128 px. A real-data
smoke run completed training, strict EMA checkpoint reload, full validation
inference, `(3868, 13)` test submission generation, and preprocessing/attention/
representation diagnostics. The smoke run used only 512 training rows and is
not a performance result. The next comparison is the complete default 64 px
CUDA run against V6 `67.118` and V7 `67.328`.

The reported V2.1 64 px CUDA run through epoch 65 had its best validation RMSE
of `70.803` at epoch 51. Training RMSE was `58.372`, temporal-attention entropy
was about `0.898`, and the image gate was about `0.411` at that checkpoint. The
image path improved the roughly `76.25` wind-only validation score, but the
one-token-per-time representation had not approached V6 or V7.

## Baseline Spatial Transformer V2.2

V2.2 tests one isolated hypothesis from the V2.1 result: the Transformer needs
the CNN's spatial representation before it is flattened. All preprocessing,
wind modeling, bounded residual fusion, loss, optimizer family, sampling, and
64 px input remain controlled.

| Property | V2.1 | V2.2 |
| --- | ---: | ---: |
| Transformer memory shape | 20 x 96 | 320 x 128 |
| Values in memory | 1,920 | 40,960 |
| Parameters | 602,640 | 669,680 |
| Timing prior default | 0.10 | 0.0 |

V2.2 uses separate spatial attention over 16 cells and temporal attention over
20 times. Its factorized encoder needs 11,520 scores per head and sample instead
of 102,400 for full attention over 320 tokens.

Local checks completed at 64 and 128 px on MPS. A real-data smoke run completed
training, EMA checkpoint reload, full validation inference, `(3868, 13)` test
submission generation, and temporal/spatial attention diagnostics. The smoke
run used only 128 training rows and is not a performance estimate. The next
decision point is the complete 64 px CUDA validation RMSE.

The complete 64 px CUDA run reached a best validation RMSE of `69.653` at epoch
26. Training continued to improve while validation flattened and then
degraded, so the larger token memory alone did not solve the generalization
problem.

## Baseline AR-Neural Transformer V2.3

V2.3 changes only the wind baseline and its residual encoder. The V2.2 image
CNN, 320-token factorized Transformer, regularization, and bounded image fusion
remain controlled.

The train rows reconstruct into 28 continuous chains and provide 10,419 unique
one-step transitions. A globally fitted recursive AR(2) forecast obtains
`75.462` validation RMSE and `73.570` chain-macro RMSE. It is stronger than the
`76.094` affine-last-wind baseline. Direct 20-lag ridge regression was about
`76.213`, while recursive differenced AR candidates were above `83`, so those
alternatives were rejected.

The neural wind path receives absolute level, level-relative history, and
first differences. Its 12 corrections are zero-initialized and bounded by
0.75 times the train AR residual scale. The complete prediction is fixed
AR(2), plus the wind correction, plus the gated V2.2 image correction. No
future row from the validation or test chains is used.

A 256-row MPS smoke run completed training, strict checkpoint reload, full
validation/test inference, submission generation, and diagnostics. Its full
validation RMSE of `75.411` is a pipeline check rather than a model result. The
next decision point is the complete default 64 px CUDA run.

## Fixed-Lag Attentive Magnitude Transformer V2.4

V2.4 tests the hypothesis that prior Transformer variants failed mainly at
unlabeled image-to-arrival time assignment. Literature on coronal-hole wind
forecasting supports a three-to-four-day transit interval and frequently uses
a fixed four-day speed lag. In this dataset, a 96-hour lag maps the 12 forecast
horizons exactly to `image_04` through `image_15`.

The cross-attention center is fixed at `96 h - forecast_horizon`, with a 12-hour
Gaussian sigma and a hard 24-hour window. Spatial attention remains global
within each 4 by 4 image grid, and temporal self-attention is restricted to a
12-hour local neighborhood. A weak central-disk prior reflects the empirical
importance of low-latitude, central-meridian coronal-hole area.

The image head now predicts at most a 15% multiplicative speed adjustment to
the AR-neural wind curve. It no longer creates a free additive forecast. The
adjustment is additionally bounded by the per-horizon AR residual scale and
starts at exactly zero.

Local verification showed mean attention ages within 0.10 hours of all fixed
source ages, temporal-attention entropy of about `0.674`, and no collapsed
features. A 256-row, four-epoch MPS smoke checkpoint completed strict reload,
full validation/test inference, `(3868, 13)` submission generation, and all
diagnostics. Its full validation RMSE was `75.426`; this is not a performance
estimate. The next decision point is the complete 64 px CUDA run, followed by
a controlled 128 px run if the fixed-lag hypothesis improves validation.

## Deep Fixed-Lag Transformer V2.5

V2.5 increases the neural contribution without relaxing V2.4's fixed 96-hour
lag or hard 24-hour attention window. Transformer depth increases from one to
two blocks and the feed-forward width from 256 to 320. The linear image scale
head becomes a 96-wide MLP, its scale limit increases from 15% to 30%, and the
initial gate increases from 18% to 40%. Small nonzero head initialization sends
gradients into both encoders on the first step.

Regularization is reduced in a controlled way: image dropout is 10%, time
masking is 2%, residual L2 is `0.0005`, and EMA decay is `0.99`. The learning
rate is `1e-4`. The parameter count is 953,520.

A five-epoch, 256-row MPS smoke run completed without divergence. Its best
checkpoint produced a full-validation RMSE of `75.242`, versus `75.426` for the
matching V2.4 pipeline check. The mean image gate was `0.400`, mean absolute
image scale was `0.506%`, and the image residual RMS was `0.869 km/s` after the
first epoch. V2.4's scale at its smoke checkpoint was about `0.005%`, so V2.5
successfully increases the deep image path's practical influence. Full CUDA
training is still required to evaluate generalization.

## Solar Hybrid V10.1 selective correction

V10's saved validation component decomposition identified a selection problem
rather than a capacity problem. AR(2) scored `75.451`, the neural wind branch
degraded it to `76.507`, propagation improved it to `68.784`, and full V10
scored `68.078`. The Transformer correction improved surge cases but degraded
fast and quiet subsets.

V10.1 therefore freezes and removes the neural wind residual from the final
sum, retains the bounded propagation term, and gates only the Transformer
correction. The gate increases with the image-only surge probability and is
suppressed for already-fast wind without surge evidence. The full CUDA result
is pending; raw and gated component variants are emitted by the V10.1
diagnostic for a controlled comparison.

## V11 Seokho V7 source map with Taeuk controls

Seokho V7 (`2df99c8`) replaces one source per frame with 160 cell sources and
supervises their longitude-time assignment using label-derived backmapping.
V11 preserves that mechanism while replacing persistence with the train-only
AR(2) baseline (`75.462` full-validation RMSE), fixing the nominal transit to
the previously tested 96-hour prior, masking the off-disk background, and
bounding the complete source contribution by the AR residual scale.

The final model deliberately excludes the free Transformer correction and
neural wind residual. Prior branch decompositions showed that those paths
lowered training error much faster than validation error, while propagation
was the repeatable generalizing component. Fast/quiet suppression from V10.1,
EMA, chain-aware metrics, time/modality masking, and strict checkpoint metadata
remain.

Local MPS verification completed with 171,537 trainable parameters. Forward,
backward, `(B,20,2,4,25)` source kernels, two-epoch real-data training, EMA
checkpoint reload, and component diagnostics passed. The reconstruction error
for `full_v11 = ar_only + propagation` was `0.000061 km/s`. The smoke run used
only 128 training and 64 validation rows, so its RMSE is a pipeline check and
not a performance estimate. The next decision point is the full 64 px CUDA
run, followed by 128 px only if the source-map gain survives complete
validation and chain/regime diagnostics.

## V11.1 Seokho-centered source-map control

The full V11 run reached `68.636` validation RMSE at epoch 11, improving its
AR(2) anchor by `6.826 km/s` but trailing the earlier V6/V7 results. Its source
propagation RMS continued from about 26 to 40 km/s while validation degraded,
showing that the AR anchor, fixed lag, cap, and selective gate did not isolate
the source-map discovery cleanly.

V11.1 therefore restores Seokho V7's model and schedule as the experimental
center. It removes AR(2), fixed 96-hour lag, AR-scaled bounds, fast/quiet
suppression, source MLP, EMA, and component regularization. The output is again
the V7 convex mixture of a learnable persistence base and the cell source-map
forecast. Only the requested soft solar-disk mask remains as a model change.

Local MPS verification passed at 64 and 128 px with 288,052 trainable
parameters. Real-data training, raw checkpoint strict reload, learnable
effective distance, `(B,20,2,4,25)` source kernels, and component diagnostics
passed. The smoke run used only 128 training and 64 validation rows and is not
a performance result. The full 64 px CUDA run should be compared directly with
Seokho V7 using the same seed and 35-epoch schedule.

## V11.2 resolution, grid, and consistency ablation

V11.2 retains three experiments around V11.1. The first fixes the augmentation
leak at the existing 64 px and `2 x 4` configuration. The retired 128 px
`2 x 4` experiment is excluded. The third jointly changes image resolution to
128 px and the longitudinal grid to eight cells. The fourth adds a 0.05
one-step overlap-consistency loss to the third configuration. Consequently,
experiments 1 and 3 do not isolate resolution from grid density.

The augmentation correction zeros physical source weight for masked times and
zeros both source weight and fusion alpha for dropped image modalities. The
dynamic grid derives longitude centers and offset bounds from cell width. The
consistency loader verified 9,579 training pairs and 1,188 validation pairs;
their image, wind, and target windows overlap exactly.

Synthetic MPS forward/backward checks passed for 64 px `2 x 4`, 128 px
`2 x 4`, and 128 px `2 x 8`. Shapes were respectively
`(B,20,2,4,25)`, `(B,20,2,4,25)`, and `(B,20,2,8,25)`. Explicit source-mask,
exact base-fallback, and paired-mask overlap checks also passed. Full CUDA
results remain pending.

## V12 LSTM soft-lag ablation

V12 restores a learned temporal sequence model after V11 showed that hard
source-speed-to-arrival routing remained weakly identified. It keeps the
competition LSTM concept but replaces the raw image front end with the soft
disk mask, signed differences, BatchNorm Inception CNN, and a 64 px `2 x 8`
longitude-preserving grid. Unlike the competition baseline, all 20 LSTM
outputs remain available to 12 horizon-specific queries.

The prediction anchor is a train-only global AR(2) fit plus a bounded neural
residual over the complete 20-step wind history. The LSTM image branch predicts
only a bounded gated correction. This directly tests image value without
requiring the image branch to relearn the wind baseline.

Three otherwise identical runs compare five lag experts, one fixed 96 hour
lag, and fully learned attention without a physical prior. All use 64 px and a
`2 x 8` grid, so the result isolates the lag assumption rather than mixing it
with resolution or model width. Full CUDA results remain pending.

## V12.1 guarded Lite U-Net token encoder

The first V12 multi-lag run reached validation RMSE `69.545` at epoch 3 while
training RMSE had already fallen to `59.354` and validation image-correction
RMS had grown to `42.438 km/s`. The process was later killed during epoch 4.
That CNN experiment is retired because the early train-validation separation
already indicates overfitting.

V12.1 keeps the AR/wind/LSTM/lag back end but replaces the image CNN with a
partial U-Net that fuses 4, 8, and 16 px features before the same `2 x 8`
token pool. The channel schedule keeps total parameters at 943,356, close to
V12's 931,116, so added spatial skips do not create a large capacity change.

The server launcher tests only guarded U-Net variants: fixed 96 hours and the
five-lag mixture. Both lower peak learning rate to `3e-5`, use batch 64 with
zero loader workers, cap image correction at 1.25 residual scales, increase
modality dropout to 0.25, use correction L2 weight 0.10, and stop after six
non-improving epochs.

Full V12.1 results confirmed that encoder replacement alone was insufficient.
The multi-lag model was best at epoch 3 with train/validation RMSE
`64.846/70.868 km/s`, chain-macro RMSE `69.363 km/s`, and correction RMS
`32.833 km/s`. Fixed 96 hours was worse at epoch 5 with train/validation RMSE
`55.454/72.045 km/s`, correction RMS `44.488 km/s`, and correction gate
`0.809`. Both are retired because the validation gap and correction growth
show that the flexible back end still bypasses transferable timing.

## V13 speed-locked U-Net timing Transformer

V13 makes Seokho's source-speed insight an architectural constraint. A partial
U-Net produces `20 x 2 x 8` source cells. Each cell's predicted speed is used
both as its forecast value and in `effective_distance / speed`; solar rotation,
image age, and that transit produce its arrival time. A small Transformer can
change attention among these candidates but cannot create an independent value
head or free residual.

Thirteen causal hindcast queries directly supervise image timing using observed
wind values. The hindcast loss decays from 0.50 to 0.10 so it initializes
mapping without dominating forecast selection. The final future prediction is
a maximum-0.5, AR-residual-capped interpolation from train-only AR(2) toward
the attention-weighted source speed. Time masking, image modality dropping,
and soft disk masking remain active.

Two runs isolate whether Seokho-style target backmapping helps the otherwise
identical speed-locked Transformer: alignment KL `0.01` versus `0.0`. Default
resolution remains 64 px because 128 px and denser grids did not improve V11.2
reliably and materially increased runtime and overfit exposure. CUDA results
are pending.

## V14 physics-guided deformable timing attention

The first V13 CUDA log showed a useful hindcast and alignment fit but almost no
future image use: validation correction RMS stayed near `1 km/s`, correction
gate near `0.024`, and validation RMSE remained effectively equal to AR(2).
V14 tests whether local timing search can connect those learned image signals
to future queries without introducing a free value path.

Each query/head starts from the eight strongest V13 physical-arrival
candidates. Seven candidates learn bounded acquisition-time and longitude
offsets while one remains an exact physical anchor. Source speed is still
shared by transit and output value, learned offsets remain causal, and a dense
reconstruction keeps the existing backmapping ablation comparable.

## V15 direct peak event branches

V15 addresses the V13 failure more directly than another attention prior. The
future target itself supplies two unambiguous value labels: its maximum speed
and, when the sequence has sufficient prominence, the 6-hour bin where that
maximum occurs. A soft one-bin timing target and prominence weighting reduce
argmax noise on flat sequences.

The heads see V14 future query features plus AR and source-speed context. Their
output can only add a bounded event-shaped correction; it cannot replace the
12-step forecast. The default ablation changes peak-time loss weight from
`0.05` to `0.10` while holding peak-value weight at `0.25` and disables the
indirect backmapping loss to avoid competing timing labels.

## V16 native longitude profile control

V16 tests whether the spatial encoders were the main source of overfitting.
No image resize, pooling, U-Net, or feature pyramid is used. Disk-masked
latitude statistics preserve all 64 native longitude columns, including dark
minima and bright maxima, before a stride-one longitude mixer and one-layer
LSTM. A fixed 96-hour prior limits timing flexibility.

The neural wind residual and indirect alignment loss are disabled. Three
matched runs compare native images, time/longitude-scrambled images, and the
exact AR(2) wind-only fallback. Native images are considered useful only when
they beat both controls on micro and chain-macro validation RMSE.
