# Transformer Review After V6

## Current V6 token path

V6 restores the V3 learnable CEA CNN while preserving the V5 forecast path.
Each of 20 observed timestamps produces a 4 by 8 grid of 72-dimensional image
tokens. A 24-dimensional causal-wind token is repeated across all 32 cells and
concatenated to form 640 tokens of width 96.

For each of four latitude bands, the current encoder flattens 20 timestamps and
8 longitudes into one sequence of length 160. Twelve horizon queries then
cross-attend to all 640 encoded tokens.

## Highest-priority improvement: factorized attention

The length-160 encoder allows every time-longitude pair to interact freely.
With only 28 independent training chains, this is more freedom than the data
can reliably constrain. A factorized block should instead apply:

1. longitude attention over 8 cells for each timestamp and latitude;
2. temporal attention over 20 timestamps for each latitude-longitude cell.

Ignoring heads and feature width, the current attention-score count per sample
is:

```text
4 * (20 * 8)^2 = 102,400
```

The factorized alternative is:

```text
20 * 4 * 8^2 + 4 * 8 * 20^2 = 17,920
```

This is 5.7 times fewer attention scores and encodes the actual separable
longitude/time structure. It is the best isolated Transformer experiment after
the V6 CNN-restoration result is available.

## Second priority: stop repeating wind tokens

Only 20 unique wind states exist, but the current model repeats each one over
32 image cells. The Transformer therefore receives 640 copies of 20 wind
states. Wind should instead be represented by 20 independent memory tokens, or
used as a FiLM modulation of image tokens before image-only attention. Forecast
queries can then attend jointly to 640 image tokens and 20 unique wind tokens.

This change removes duplicate evidence and prevents the wind subspace from
appearing once for every spatial location.

## Third priority: bounded image correction

The V5 validation image-residual RMS varied from 58 to 81 km/s while its wind
RMSE stayed near 76 km/s. The unconstrained image head caused the large
validation swings. A horizon-scaled correction is safer:

```text
image_residual[h] = scale[h] * tanh(raw[h] / scale[h])
```

The scale can be initialized from training residual standard deviation and
optionally multiplied by a learned sigmoid gate. This preserves useful image
corrections while limiting catastrophic extrapolation.

## Later experiments

- Add a learnable relative attention bias initialized to favor longitudinal
  displacement consistent with solar rotation.
- Impose smoothness across the 12 horizon queries with a small second-difference
  penalty or a lightweight 1D horizon decoder.
- Save raw validation predictions from every model and fit convex ensemble
  weights out of fold; do not rank models only from scalar RMSE.

## Experiment order

1. V6: restore the V3 CNN only.
2. V7: replace length-160 axial attention with factorized longitude/temporal
   attention, keeping every other V6 setting fixed.
3. V8: separate the 20 wind tokens from spatial image memory.
4. Add bounded residual gating only after the first three ablations establish
   which representation generalizes.

## V7 implementation status

The factorized-attention experiment is implemented as V7. Longitude and time
passes share one MHA kernel, keeping the comparison parameter-controlled:

- V6 parameters: 203,197;
- V7 parameters: 203,389;
- parameter difference: 192;
- V7 attention-score count: 17,920, matching the calculation above.

V7 has passed 128 px MPS forward/backward and the complete local smoke pipeline.
Its CUDA validation RMSE must be compared with V6 before proceeding to separate
wind memory tokens.
