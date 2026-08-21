import os
from pathlib import Path

import pandas as pd

from model_solar_lstm_v12 import FILE_STEM


EXPERIMENTS = (
    "exp1_multilag",
    "exp2_fixed96",
    "exp3_learned_only",
)


def value(row, column):
    return row[column] if column in row.index else float("nan")


def main():
    root = Path(
        os.getenv(
            "V12_ABLATION_ROOT",
            "/home/jovyan/outputs/solar_lag_lstm_v12_ablation_seed777",
        )
    )
    rows = []
    for experiment in EXPERIMENTS:
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
                "val_wind_base_rmse_km_s": value(
                    best, "val_wind_base_rmse_km_s"
                ),
                "val_image_correction_rms_km_s": value(
                    best, "val_image_correction_rms_km_s"
                ),
                "val_lag_alignment_kl": value(
                    best, "val_lag_alignment_kl"
                ),
                "val_attention_delay_h": value(
                    best, "val_attention_delay_h"
                ),
                "val_attention_entropy": value(
                    best, "val_attention_entropy"
                ),
                "val_expected_lag_h": value(best, "val_expected_lag_h"),
                "val_correction_gate": value(best, "val_correction_gate"),
                "seconds": value(best, "seconds"),
            }
        )
    if not rows:
        raise FileNotFoundError(f"no V12 histories found below {root}")
    summary = pd.DataFrame(rows).sort_values("val_rmse_km_s")
    summary_path = root / f"{FILE_STEM}_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"saved: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
