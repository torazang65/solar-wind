import os
from pathlib import Path

import pandas as pd

from model_solar_source_map_v11_2 import FILE_STEM


EXPERIMENTS = (
    "exp1_64_2x4_maskfix",
    "exp2_128_2x4_maskfix",
    "exp3_128_2x8_maskfix",
    "exp4_128_2x8_consistency",
)


def value(row, column):
    return row[column] if column in row.index else float("nan")


def main():
    root = Path(
        os.getenv(
            "V112_ABLATION_ROOT",
            "/home/jovyan/outputs/solar_source_map_v11_2_ablation_seed777",
        )
    )
    rows = []
    for experiment in EXPERIMENTS:
        history_path = root / experiment / f"{FILE_STEM}_history.csv"
        if not history_path.exists():
            raise FileNotFoundError(f"missing experiment history: {history_path}")
        history = pd.read_csv(history_path)
        if history.empty or "val_rmse_km_s" not in history:
            raise ValueError(f"invalid experiment history: {history_path}")
        best = history.loc[history["val_rmse_km_s"].idxmin()]
        rows.append(
            {
                "experiment": experiment,
                "best_epoch": int(best["epoch"]),
                "train_rmse_km_s": value(best, "train_rmse_km_s"),
                "val_rmse_km_s": value(best, "val_rmse_km_s"),
                "val_chain_macro_rmse_km_s": value(
                    best, "val_chain_macro_rmse_km_s"
                ),
                "val_base_rmse_km_s": value(best, "val_base_rmse_km_s"),
                "val_hindcast_rmse_km_s": value(
                    best, "val_hindcast_rmse_km_s"
                ),
                "val_alignment_kl": value(best, "val_alignment_kl"),
                "val_surge_auroc": value(best, "val_surge_auroc"),
                "val_propagation_rms_km_s": value(
                    best, "val_propagation_rms_km_s"
                ),
                "val_consistency_rmse_km_s": value(
                    best, "val_consistency_rmse_km_s"
                ),
                "seconds": value(best, "seconds"),
            }
        )
    summary = pd.DataFrame(rows).sort_values("val_rmse_km_s")
    summary_path = root / f"{FILE_STEM}_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"saved: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
