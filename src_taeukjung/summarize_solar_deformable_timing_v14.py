import os
from pathlib import Path

import pandas as pd

from model_solar_deformable_timing_v14 import FILE_STEM


EXPERIMENTS = (
    "v14_deformable_full",
    "v14_deformable_no_backmapping",
)


def value(row, column):
    return row[column] if column in row.index else float("nan")


def main():
    root = Path(
        os.getenv(
            "V14_ABLATION_ROOT",
            "/home/jovyan/outputs/solar_deformable_timing_v14_seed777",
        )
    )
    rows = []
    experiments = os.getenv("V14_EXPERIMENTS", " ".join(EXPERIMENTS)).split()
    for experiment in experiments:
        history_path = root / experiment / f"{FILE_STEM}_history.csv"
        if not history_path.exists():
            print(f"skipping missing history: {history_path}")
            continue
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
                "val_ar_base_rmse_km_s": value(
                    best, "val_ar_base_rmse_km_s"
                ),
                "val_correction_rms_km_s": value(
                    best, "val_image_correction_rms_km_s"
                ),
                "val_hindcast_rmse_km_s": value(
                    best, "val_hindcast_rmse_km_s"
                ),
                "val_backmapping_kl": value(best, "val_backmapping_kl"),
                "val_attention_delay_h": value(
                    best, "val_attention_delay_h"
                ),
                "val_sparse_attention_entropy": value(
                    best, "val_sparse_attention_entropy"
                ),
                "val_deform_time_offset_h": value(
                    best, "val_deform_time_offset_h"
                ),
                "val_deform_longitude_offset_cells": value(
                    best, "val_deform_longitude_offset_cells"
                ),
                "val_correction_gate": value(best, "val_correction_gate"),
                "seconds": value(best, "seconds"),
            }
        )
    if not rows:
        raise FileNotFoundError(f"no V14 histories found below {root}")
    summary = pd.DataFrame(rows).sort_values("val_rmse_km_s")
    summary_path = root / f"{FILE_STEM}_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"saved: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
