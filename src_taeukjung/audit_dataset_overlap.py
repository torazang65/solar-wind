import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IMAGE_COLUMNS = [f"image_{index:02d}" for index in range(20)]
WIND_COLUMNS = [f"wind_{index:02d}" for index in range(20)]
TARGET_COLUMNS = [f"target_{index:02d}" for index in range(12)]


def infer_temporal_chains(inputs):
    sequences = inputs[IMAGE_COLUMNS].astype(str).to_numpy()
    prefix_to_row = {}
    for row_index, sequence in enumerate(sequences):
        prefix = tuple(sequence[:-1])
        if prefix in prefix_to_row:
            raise ValueError("image-window prefixes are not unique")
        prefix_to_row[prefix] = row_index

    successor = np.full(len(sequences), -1, dtype=np.int64)
    predecessor = np.full(len(sequences), -1, dtype=np.int64)
    for row_index, sequence in enumerate(sequences):
        next_row = prefix_to_row.get(tuple(sequence[1:]))
        if next_row is None:
            continue
        if predecessor[next_row] >= 0:
            raise ValueError("a row has multiple temporal predecessors")
        successor[row_index] = next_row
        predecessor[next_row] = row_index

    chains = []
    seen = np.zeros(len(sequences), dtype=bool)
    for start in np.flatnonzero(predecessor < 0):
        chain = []
        row_index = int(start)
        while row_index >= 0:
            if seen[row_index]:
                raise ValueError("cycle or merged temporal chain detected")
            seen[row_index] = True
            chain.append(row_index)
            row_index = int(successor[row_index])
        chains.append(tuple(chain))
    if not np.all(seen):
        raise ValueError("cyclic temporal chains are unsupported")
    return successor, tuple(chains)


def follow_successor(successor, steps):
    destination = np.arange(len(successor), dtype=np.int64)
    available = np.ones(len(successor), dtype=bool)
    for _ in range(steps):
        active = available & (destination >= 0)
        destination[active] = successor[destination[active]]
        available &= destination >= 0
    return destination, available


def fit_latest_wind_baseline(train_inputs, train_targets):
    latest = train_inputs[WIND_COLUMNS[-1]].to_numpy(np.float64)
    design = np.column_stack([latest, np.ones(len(latest))])
    coefficients = np.linalg.lstsq(design, train_targets, rcond=None)[0]
    return coefficients[0], coefficients[1]


def predict_latest_wind_baseline(inputs, slope, intercept):
    latest = inputs[WIND_COLUMNS[-1]].to_numpy(np.float64)[:, None]
    return latest * slope[None, :] + intercept[None, :]


def audit_split(split, inputs, targets=None):
    successor, chains = infer_temporal_chains(inputs)
    lengths = np.asarray([len(chain) for chain in chains], dtype=np.int64)
    unique_images = len(pd.unique(inputs[IMAGE_COLUMNS].to_numpy().ravel()))
    summary = {
        "split": split,
        "rows": int(len(inputs)),
        "chains": int(len(chains)),
        "chain_lengths": lengths.tolist(),
        "minimum_chain_length": int(lengths.min()),
        "maximum_chain_length": int(lengths.max()),
        "unique_images": int(unique_images),
        "sliding_window_identity": bool(
            unique_images == len(inputs) + 19 * len(chains)
        ),
    }

    rows = []
    latest_wind = inputs[WIND_COLUMNS[-1]].to_numpy(np.float64)
    for horizon in range(1, 13):
        destination, available = follow_successor(successor, horizon)
        row = {
            "split": split,
            "horizon_hours": horizon * 6,
            "covered_rows": int(available.sum()),
            "total_rows": int(len(inputs)),
            "coverage": float(available.mean()),
        }
        if targets is not None:
            errors = (
                latest_wind[destination[available]]
                - targets[available, horizon - 1]
            )
            row.update(
                {
                    "exact_match_rate": float(np.mean(errors == 0.0)),
                    "successor_wind_rmse_km_s": float(
                        np.sqrt(np.mean(errors**2))
                    ),
                }
            )
        rows.append(row)
    summary["mean_target_value_coverage"] = float(
        np.mean([row["coverage"] for row in rows])
    )
    return summary, pd.DataFrame(rows), successor


def stitch_observed_future_wind(inputs, predictions, successor):
    stitched = np.asarray(predictions, dtype=np.float64).copy()
    covered = np.zeros(stitched.shape, dtype=bool)
    latest_wind = inputs[WIND_COLUMNS[-1]].to_numpy(np.float64)
    for horizon in range(1, 13):
        destination, available = follow_successor(successor, horizon)
        stitched[available, horizon - 1] = latest_wind[destination[available]]
        covered[available, horizon - 1] = True
    return stitched, covered


def make_figure(output_path, summaries, overlap, baseline_rmse, stitched_rmse):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for summary in summaries:
        axes[0].hist(
            summary["chain_lengths"],
            bins=min(12, len(summary["chain_lengths"])),
            alpha=0.55,
            label=summary["split"],
        )
    axes[0].set_title("Recovered temporal chains")
    axes[0].set_xlabel("Rows per chain")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    for split, frame in overlap.groupby("split", sort=False):
        axes[1].plot(
            frame.horizon_hours,
            frame.coverage * 100.0,
            marker="o",
            label=split,
        )
    axes[1].set_title("Targets present as later input wind")
    axes[1].set_xlabel("Forecast horizon (hours)")
    axes[1].set_ylabel("Coverage (%)")
    axes[1].set_ylim(85, 101)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    horizons = np.arange(1, 13) * 6
    axes[2].plot(horizons, baseline_rmse, marker="o", label="linear baseline")
    axes[2].plot(horizons, stitched_rmse, marker="o", label="chain stitched")
    axes[2].set_title("Validation RMSE")
    axes[2].set_xlabel("Forecast horizon (hours)")
    axes[2].set_ylabel("RMSE (km/s)")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit deterministic overlap between shuffled solar-wind windows."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("public_dataset/competition_dataset_6h"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/data_audit_taeukjung"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = {
        split: pd.read_csv(args.data_root / split / "inputs.csv")
        for split in ("train", "validation", "test")
    }
    target_frames = {
        split: pd.read_csv(args.data_root / split / "targets.csv")
        for split in ("train", "validation")
    }
    targets = {
        split: target_frames[split][TARGET_COLUMNS].to_numpy(np.float64)
        for split in target_frames
    }
    for split in target_frames:
        if not np.array_equal(
            inputs[split].sample_id.to_numpy(),
            target_frames[split].sample_id.to_numpy(),
        ):
            raise ValueError(f"{split} input and target IDs are not aligned")

    summaries = []
    overlap_frames = []
    successors = {}
    for split in ("train", "validation", "test"):
        summary, frame, successor = audit_split(
            split, inputs[split], targets.get(split)
        )
        summaries.append(summary)
        overlap_frames.append(frame)
        successors[split] = successor
    overlap = pd.concat(overlap_frames, ignore_index=True)

    slope, intercept = fit_latest_wind_baseline(inputs["train"], targets["train"])
    validation_baseline = predict_latest_wind_baseline(
        inputs["validation"], slope, intercept
    )
    validation_stitched, covered = stitch_observed_future_wind(
        inputs["validation"], validation_baseline, successors["validation"]
    )
    validation_target = targets["validation"]
    baseline_error = validation_baseline - validation_target
    stitched_error = validation_stitched - validation_target
    baseline_horizon_rmse = np.sqrt(np.mean(baseline_error**2, axis=0))
    stitched_horizon_rmse = np.sqrt(np.mean(stitched_error**2, axis=0))

    result = {
        "splits": {summary["split"]: summary for summary in summaries},
        "latest_wind_linear_baseline": {
            "slope": slope.tolist(),
            "intercept": intercept.tolist(),
            "validation_rmse_km_s": float(np.sqrt(np.mean(baseline_error**2))),
        },
        "validation_chain_stitch": {
            "covered_target_values": int(covered.sum()),
            "total_target_values": int(covered.size),
            "coverage": float(covered.mean()),
            "overall_rmse_km_s": float(np.sqrt(np.mean(stitched_error**2))),
            "fallback_only_rmse_km_s": float(
                np.sqrt(np.mean(stitched_error[~covered] ** 2))
            ),
            "horizon_rmse_km_s": stitched_horizon_rmse.tolist(),
        },
        "warning": (
            "Split-wide successor-wind stitching is transductive. Confirm the "
            "competition rules before using it for a test submission."
        ),
    }

    overlap.to_csv(args.output_dir / "temporal_overlap_by_horizon.csv", index=False)
    (args.output_dir / "dataset_overlap_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    make_figure(
        args.output_dir / "temporal_overlap_audit.png",
        summaries,
        overlap,
        baseline_horizon_rmse,
        stitched_horizon_rmse,
    )

    print(json.dumps(result, indent=2))
    print(f"saved: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
