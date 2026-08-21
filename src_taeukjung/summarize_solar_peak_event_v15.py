import os
from pathlib import Path

import pandas as pd

from model_solar_peak_event_v15 import FILE_STEM


EXPERIMENTS = ("v15_peak_joint", "v15_peak_time_strong")


def value(row, column):
    return row[column] if column in row.index else float("nan")


def main():
    root = Path(
        os.getenv(
            "V15_ABLATION_ROOT",
            "/home/jovyan/outputs/solar_peak_event_v15_seed777",
        )
    )
    experiments = os.getenv("V15_EXPERIMENTS", " ".join(EXPERIMENTS)).split()
    rows = []
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
                "val_peak_time_mae_h": value(best, "val_peak_time_mae_h"),
                "val_peak_time_within_6h": value(
                    best, "val_peak_time_within_6h"
                ),
                "val_peak_value_rmse_km_s": value(
                    best, "val_peak_value_rmse_km_s"
                ),
                "val_predicted_peak_hour": value(
                    best, "val_predicted_peak_hour"
                ),
                "val_predicted_peak_value_kms": value(
                    best, "val_predicted_peak_value_kms"
                ),
                "val_peak_event_correction_rms_kms": value(
                    best, "val_peak_event_correction_rms_kms"
                ),
                "val_correction_rms_km_s": value(
                    best, "val_image_correction_rms_km_s"
                ),
                "seconds": value(best, "seconds"),
            }
        )
    if not rows:
        raise FileNotFoundError(f"no V15 histories found below {root}")
    summary = pd.DataFrame(rows).sort_values("val_rmse_km_s")
    summary_path = root / f"{FILE_STEM}_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"saved: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
