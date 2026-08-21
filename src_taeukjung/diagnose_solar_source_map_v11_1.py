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
from inference_solar_source_map_v11_1 import load_best_model, predict
from model_solar_source_map_v11_1 import FILE_STEM


def limited_indexes(indexes):
    limit = int(os.getenv("MAX_VAL_SAMPLES", "0"))
    if limit <= 0 or limit >= len(indexes):
        return indexes
    return indexes[:limit]


def rmse(actual, prediction, rows=None):
    if rows is not None:
        actual = actual[rows]
        prediction = prediction[rows]
    return float(np.sqrt(np.mean((prediction - actual) ** 2)))


def main():
    model, checkpoint = load_best_model()
    val_chains = infer_temporal_chains(val_inputs, IMAGE_COLUMNS)
    selected_val_index = limited_indexes(val_index)
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
    base_prediction = components["base"]
    source_prediction = components["source_prediction"]
    reconstructed = base_prediction + components["propagation_residual"]
    variants = {
        "seokho_base": base_prediction,
        "source_only": source_prediction,
        "base_plus_source_map": reconstructed,
        "full_v11_1": full_prediction,
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
        base_group_rmse = rmse(actual, base_prediction, mask)
        for variant_name, prediction in variants.items():
            value = rmse(actual, prediction, mask)
            rows.append(
                {
                    "group": group_name,
                    "samples": int(mask.sum()),
                    "variant": variant_name,
                    "rmse_km_s": value,
                    "gain_vs_base_km_s": base_group_rmse - value,
                }
            )
    component_frame = pd.DataFrame(rows)
    component_path = OUTPUT_DIR / f"{FILE_STEM}_component_gain.csv"
    component_frame.to_csv(component_path, index=False)

    horizon_rows = []
    for horizon_index in range(12):
        row = {"horizon_hours": (horizon_index + 1) * 6}
        for name, prediction in variants.items():
            row[f"{name}_rmse_km_s"] = rmse(
                actual[:, horizon_index], prediction[:, horizon_index]
            )
        horizon_rows.append(row)
    horizon_path = OUTPUT_DIR / f"{FILE_STEM}_horizon_decomposition.csv"
    pd.DataFrame(horizon_rows).to_csv(horizon_path, index=False)

    chain_rows = []
    for chain_id in np.unique(chain_ids):
        mask = chain_ids == chain_id
        chain_rows.append(
            {
                "chain_id": int(chain_id),
                "samples": int(mask.sum()),
                "full_rmse_km_s": rmse(actual, full_prediction, mask),
                "base_rmse_km_s": rmse(actual, base_prediction, mask),
            }
        )
    chain_path = OUTPUT_DIR / f"{FILE_STEM}_chain_metrics.csv"
    pd.DataFrame(chain_rows).to_csv(chain_path, index=False)

    reconstruction_error = float(np.max(np.abs(full_prediction - reconstructed)))
    if reconstruction_error > 1e-3:
        raise RuntimeError(
            f"V11.1 component reconstruction failed: {reconstruction_error}"
        )
    print(f"checkpoint_epoch={checkpoint['epoch']}")
    print(f"component_reconstruction_max_abs_km_s={reconstruction_error:.6f}")
    for row in component_frame[component_frame.group == "all"].itertuples(
        index=False
    ):
        print(
            f"{row.variant}: rmse={row.rmse_km_s:.3f} "
            f"gain_vs_base={row.gain_vs_base_km_s:+.3f}"
        )
    print(f"saved: {component_path.resolve()}")
    print(f"saved: {horizon_path.resolve()}")
    print(f"saved: {chain_path.resolve()}")


if __name__ == "__main__":
    main()
