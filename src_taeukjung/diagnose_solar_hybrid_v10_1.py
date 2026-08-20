import os

import numpy as np
import pandas as pd

from chain_sampling import (
    ChainAwareSolarWindDataset,
    infer_temporal_chains,
    make_chain_loader,
)
from config import OUTPUT_DIR
from dataset import (
    IMAGE_COLUMNS,
    WIND_COLUMNS,
    val_image_array,
    val_image_index,
    val_index,
    val_inputs,
    val_targets,
)
from inference_solar_hybrid_v10 import load_best_model, predict
from model_solar_hybrid_v10_1 import (
    ARCHITECTURE_NAME,
    FILE_STEM,
    SolarWindSelectiveHybridV101,
)


def limited_indexes(indexes, environment_name):
    limit = int(os.getenv(environment_name, "0"))
    if limit <= 0 or limit >= len(indexes):
        return indexes
    return indexes[:limit]


def rmse(actual, prediction, rows=None):
    if rows is not None:
        actual = actual[rows]
        prediction = prediction[rows]
    return float(np.sqrt(np.mean((prediction - actual) ** 2)))


def main():
    model, checkpoint = load_best_model(
        SolarWindSelectiveHybridV101,
        ARCHITECTURE_NAME,
        FILE_STEM,
    )
    val_chains = infer_temporal_chains(val_inputs, IMAGE_COLUMNS)
    selected_val_index = limited_indexes(val_index, "MAX_VAL_SAMPLES")
    dataset = ChainAwareSolarWindDataset(
        val_image_array,
        val_image_index,
        val_inputs,
        selected_val_index,
        val_targets,
        temporal_chains=val_chains,
    )
    loader = make_chain_loader(dataset, training=False)
    full_prediction, _, chain_ids, components = predict(
        model, loader, return_components=True
    )
    actual = np.asarray(val_targets[selected_val_index], dtype=np.float64)
    ar_prediction = components["ar_baseline"]
    propagation = components["propagation_residual"]
    raw_correction = components["raw_correction"]
    gated_correction = components["correction"]
    correction_gate = components["correction_gate"]
    surge_probability = components["surge_probability"]

    reconstructed_full = ar_prediction + propagation + gated_correction
    if not np.allclose(full_prediction, reconstructed_full, atol=2e-3, rtol=1e-5):
        maximum_error = float(np.max(np.abs(full_prediction - reconstructed_full)))
        raise RuntimeError(
            f"V10.1 component reconstruction failed: max_error={maximum_error:.6f}"
        )

    variants = {
        "ar_only": ar_prediction,
        "ar_plus_propagation": ar_prediction + propagation,
        "ar_plus_raw_transformer": ar_prediction + raw_correction,
        "ar_plus_gated_transformer": ar_prediction + gated_correction,
        "ar_plus_propagation_raw_transformer": (
            ar_prediction + propagation + raw_correction
        ),
        "full_v10_1": full_prediction,
    }

    latest_wind = val_inputs.iloc[selected_val_index][WIND_COLUMNS[-1]].to_numpy(
        np.float64
    )
    future_change = actual.max(axis=1) - latest_wind
    groups = {
        "all": np.ones(len(actual), dtype=bool),
        "slow": latest_wind < 400.0,
        "mid": (latest_wind >= 400.0) & (latest_wind < 550.0),
        "fast": latest_wind >= 550.0,
        "surge": future_change > 100.0,
        "quiet": future_change <= 100.0,
    }
    rows = []
    for group_name, mask in groups.items():
        if not np.any(mask):
            continue
        ar_group_rmse = rmse(actual, ar_prediction, mask)
        mean_gate = float(correction_gate[mask].mean())
        mean_surge_probability = float(surge_probability[mask].mean())
        for variant_name, prediction in variants.items():
            value = rmse(actual, prediction, mask)
            rows.append(
                {
                    "group": group_name,
                    "samples": int(mask.sum()),
                    "variant": variant_name,
                    "rmse_km_s": value,
                    "gain_vs_ar_km_s": ar_group_rmse - value,
                    "mean_correction_gate": mean_gate,
                    "mean_surge_probability": mean_surge_probability,
                }
            )
    component_frame = pd.DataFrame(rows)
    component_path = OUTPUT_DIR / f"{FILE_STEM}_component_gain.csv"
    component_frame.to_csv(component_path, index=False)

    horizon_rows = []
    for horizon_index in range(12):
        row = {
            "horizon_hours": (horizon_index + 1) * 6,
            "mean_correction_gate": float(correction_gate[:, horizon_index].mean()),
            "mean_surge_probability": float(
                surge_probability[:, horizon_index].mean()
            ),
        }
        for name, prediction in variants.items():
            row[f"{name}_rmse_km_s"] = rmse(
                actual[:, horizon_index], prediction[:, horizon_index]
            )
        horizon_rows.append(row)
    horizon_path = OUTPUT_DIR / f"{FILE_STEM}_horizon_decomposition.csv"
    pd.DataFrame(horizon_rows).to_csv(horizon_path, index=False)

    error = (full_prediction - actual) ** 2
    chain_rows = []
    for chain_id in np.unique(chain_ids):
        mask = chain_ids == chain_id
        chain_rows.append(
            {
                "chain_id": int(chain_id),
                "samples": int(mask.sum()),
                "full_rmse_km_s": float(np.sqrt(np.mean(error[mask]))),
                "ar_rmse_km_s": rmse(actual, ar_prediction, mask),
                "mean_correction_gate": float(correction_gate[mask].mean()),
                "mean_surge_probability": float(
                    surge_probability[mask].mean()
                ),
            }
        )
    chain_path = OUTPUT_DIR / f"{FILE_STEM}_chain_metrics.csv"
    pd.DataFrame(chain_rows).to_csv(chain_path, index=False)

    print(f"checkpoint_epoch={checkpoint['epoch']}")
    for group_name in groups:
        group_rows = component_frame[component_frame.group == group_name]
        if group_rows.empty:
            continue
        print(f"\n[{group_name}] samples={int(group_rows.iloc[0]['samples'])}")
        for row in group_rows.itertuples(index=False):
            print(
                f"{row.variant}: rmse={row.rmse_km_s:.3f} "
                f"gain_vs_ar={row.gain_vs_ar_km_s:+.3f}"
            )
        print(
            f"gate={group_rows.iloc[0]['mean_correction_gate']:.3f} "
            f"surge_probability="
            f"{group_rows.iloc[0]['mean_surge_probability']:.3f}"
        )
    print(f"saved: {component_path.resolve()}")
    print(f"saved: {horizon_path.resolve()}")
    print(f"saved: {chain_path.resolve()}")


if __name__ == "__main__":
    main()
