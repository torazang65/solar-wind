import os
from pathlib import Path

import pandas as pd

from model_solar_transport_fusion_v17 import FILE_STEM


EXPERIMENTS = ("v17_native", "v17_scrambled", "v17_no_pretrain")


def value(row, column):
    return row[column] if column in row.index else float("nan")


def main():
    root = Path(
        os.getenv(
            "V17_ABLATION_ROOT",
            "/home/jovyan/outputs/solar_transport_fusion_v17_seed777",
        )
    )
    experiments = os.getenv("V17_EXPERIMENTS", " ".join(EXPERIMENTS)).split()
    rows = []
    for experiment in experiments:
        history_path = root / experiment / f"{FILE_STEM}_history.csv"
        if not history_path.exists():
            print(f"skipping missing history: {history_path}")
            continue
        history = pd.read_csv(history_path)
        forecast_history = history[history["stage"] != "transport"]
        if forecast_history.empty:
            raise ValueError(f"no forecast stage in history: {history_path}")
        best = forecast_history.loc[forecast_history["val_rmse_km_s"].idxmin()]
        rows.append(
            {
                "experiment": experiment,
                "best_epoch": int(best["epoch"]),
                "best_stage": best["stage"],
                "train_rmse_km_s": value(best, "train_rmse_km_s"),
                "val_rmse_km_s": value(best, "val_rmse_km_s"),
                "val_chain_macro_rmse_km_s": value(
                    best, "val_chain_macro_rmse_km_s"
                ),
                "val_ar_base_rmse_km_s": value(best, "val_ar_base_rmse_km_s"),
                "val_transport_hindcast_rmse_km_s": value(
                    best, "val_transport_hindcast_rmse_km_s"
                ),
                "val_transport_future_rmse_km_s": value(
                    best, "val_transport_future_rmse_km_s"
                ),
                "val_correction_rms_km_s": value(
                    best, "val_image_correction_rms_km_s"
                ),
                "val_expected_delay_h": value(
                    best, "val_transport_expected_delay_h"
                ),
                "val_expert_entropy": value(best, "val_expert_entropy"),
                "seconds": value(best, "seconds"),
            }
        )
    if not rows:
        raise FileNotFoundError(f"no V17 histories found below {root}")
    summary = pd.DataFrame(rows).sort_values("val_rmse_km_s")
    summary_path = root / f"{FILE_STEM}_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"saved: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
