# Solar Wind Dataset Overlap Audit

## Executive finding

The V2.2 run is numerically healthy. Its best validation RMSE, 69.653 km/s at
epoch 26, is not evidence of a broken Transformer. The larger spatial memory
improved V2.1's 70.803 km/s result by 1.150 km/s, but the training data contains
far fewer independent sequences than its 9,607 rows suggest.

Every row is a one-step sliding window over a longer time series. Shuffled row
order and randomized image filenames hide the ordering, but adjacent rows are
recovered exactly because 19 of their 20 image filenames overlap.

| Split | Rows | Recovered chains | Chain lengths | Unique images |
| --- | ---: | ---: | --- | ---: |
| train | 9,607 | 28 | 14-961 | 10,139 |
| validation | 1,199 | 11 | all 109 | 1,408 |
| test | 3,868 | 13 | 140-357 | 4,115 |

For every split, the identity below holds exactly:

```text
unique images = rows + 19 * chains
```

Consequently, a unique timestamp is reused by up to 20 nominal training rows.
Random row shuffling does not create 9,607 independent solar regimes.

## Exact future-wind overlap

For a row at chain position `p`, its target at horizon `h` is exactly the last
wind observation of the row at position `p + h`:

```text
target_(h-1)[p] == wind_19[p+h], h = 1..12
```

This equality was checked against every available train and validation pair.
The exact-match rate is 100 percent and the numerical RMSE is 0.0 km/s.

Validation coverage decreases from 99.083 percent at 6 hours to 88.991 percent
at 72 hours. Across all 14,388 validation target values, 13,530 values, or
94.037 percent, are already present in another validation input row. Test has
longer chains: 97.816 percent of its target positions have a corresponding
later input row.

As a diagnostic only, replacing covered validation predictions with the later
row's `wind_19` reduces the train-fitted latest-wind linear baseline from
76.094 to 32.436 km/s overall. The remaining 5.963 percent are chain-edge
targets and retain the baseline prediction. This is split-wide transductive
postprocessing, not ordinary independent-row forecasting.

Do not connect this operation to the submitted inference path until the
organizers explicitly confirm that using other test rows to reconstruct a
sample's future is allowed. The local competition files contain no rule that
settles this question, and no relevant organizer statement was found in the
public Slack history.

## What the V2.2 log says

The model learned useful image information:

- validation RMSE improved monotonically from 76.175 to 69.653 by epoch 26;
- the image correction grew from 0.18 to about 43 km/s RMS;
- temporal attention entropy fell from 1.0 to about 0.88;
- spatial attention entropy fell from 1.0 to about 0.69 at the best epoch and
  continued toward 0.61 afterward.

The final item is the main warning. More spatial tokens let the model select
specific locations, but the same solar frames recur in many overlapping rows.
After epoch 26, train RMSE keeps falling while validation stays near 69.7 and
then worsens. The extra representation is being used increasingly to memorize
the 28 training chains rather than to learn a split-stable image relation.

V2.2 is therefore not under-capacity. Increasing `d_model`, token count, or
encoder depth again is unlikely to solve the generalization gap.

## Image inspection

The original files are 512 x 512 pseudo-colored 193 A and 211 A PNGs. Converting
them to grayscale and resizing to 64 x 64 is working as intended. Aggregate
on-disk intensity distributions are similar between train and validation.

The radius-0.49 mask removes the corners and the outermost off-limb corona. It
does not erase the main solar disk or central coronal-hole morphology. It is
therefore not the primary cause of the 69 km/s plateau, although off-limb CME
information outside the mask cannot contribute.

## Evaluation implications

Two distinct experiments must no longer be mixed:

1. **Independent-row forecast:** only the 20 image/wind observations in the
   row are available. V2.2's 69.653 km/s is a valid result under this protocol.
2. **Split-wide transductive forecast:** all shuffled validation or test rows
   are jointly available. Most targets can be reconstructed deterministically
   from successor rows, and model quality matters primarily at chain ends.

For honest architecture comparisons, report independent-row RMSE and also a
chain-edge metric where the future timestamp does not occur in another input.
For any leaderboard submission, obtain a ruling before using split-wide future
row reconstruction.

## Reproduction

From the repository root:

```bash
python src_taeukjung/audit_dataset_overlap.py \
  --data-root ../dev/public_dataset/competition_dataset_6h \
  --output-dir ../dev/outputs/data_audit_taeukjung
```

The script writes a JSON summary, horizon-level CSV, and
`temporal_overlap_audit.png`.
