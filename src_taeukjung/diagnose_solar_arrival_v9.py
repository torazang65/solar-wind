import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from chain_sampling import (
    ChainAwareSolarWindDataset,
    infer_temporal_chains,
    make_chain_loader,
)
from config import *
from dataset import (
    IMAGE_COLUMNS,
    val_image_array,
    val_image_index,
    val_index,
    val_inputs,
    val_targets,
)
from inference_solar_physics_v5 import load_best_model
from model_solar_arrival_v9 import SolarWindArrivalTCNV9


FILE_STEM = "solar_arrival_v9"
ANALYSIS_DIR = OUTPUT_DIR / f"{FILE_STEM}_diagnostics"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
MAX_BATCHES = int(os.getenv("SOLAR_V9_DIAGNOSTIC_MAX_BATCHES", "0"))
EVENT_THRESHOLD_KM_S = float(os.getenv("SOLAR_V9_EVENT_THRESHOLD_KM_S", "100"))


def rmse(error):
    return float(np.sqrt(np.mean(np.square(error))))


def representation_statistics(tokens):
    samples, steps, dimensions = tokens.shape
    flat = tokens.reshape(samples * steps, dimensions).astype(np.float64)
    centered = flat - flat.mean(axis=0, keepdims=True)
    feature_std = centered.std(axis=0)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = np.square(singular)
    probability = energy / max(float(energy.sum()), 1e-12)
    effective_rank = float(
        np.exp(-np.sum(probability * np.log(np.maximum(probability, 1e-12))))
    )

    normalized = tokens / np.maximum(
        np.linalg.norm(tokens, axis=-1, keepdims=True), 1e-8
    )
    adjacent_cosine = float(
        np.mean(np.sum(normalized[:, 1:] * normalized[:, :-1], axis=-1))
    )
    random_cosine = float(
        np.mean(np.sum(normalized[:, -1] * np.roll(normalized[:, -1], 1, axis=0), axis=-1))
    )
    return {
        "feature_dimension": int(dimensions),
        "effective_rank": effective_rank,
        "median_feature_std": float(np.median(feature_std)),
        "minimum_feature_std": float(feature_std.min()),
        "collapsed_feature_fraction": float(np.mean(feature_std < 1e-3)),
        "adjacent_time_cosine": adjacent_cosine,
        "random_sample_cosine": random_cosine,
    }, singular


def save_gate_plots(mean_weights, horizon_frame, source_frame, singular_values):
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    image = ax.imshow(mean_weights, aspect="auto", cmap="viridis")
    ax.set_xticks(range(0, 20, 2))
    ax.set_xticklabels([f"-{(19 - index) * 6}h" for index in range(0, 20, 2)])
    ax.set_yticks(range(12))
    ax.set_yticklabels([f"+{(index + 1) * 6}h" for index in range(12)])
    ax.set_xlabel("image observation age")
    ax.set_ylabel("forecast horizon")
    ax.set_title("V9 mean learned arrival gate")
    fig.colorbar(image, ax=ax, label="attention weight")
    fig.savefig(ANALYSIS_DIR / "arrival_gate_mean.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    ax.plot(
        horizon_frame.horizon_hours,
        horizon_frame.expected_image_age_hours,
        marker="o",
        label="learned expected image age",
    )
    ax.set_xlabel("forecast horizon (hours)")
    ax.set_ylabel("image age (hours before forecast origin)")
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(ANALYSIS_DIR / "arrival_expected_age.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    ax.plot(
        source_frame.image_age_hours,
        source_frame.mean_transit_hours,
        marker="o",
        label="predicted transit",
    )
    ax.plot(
        source_frame.image_age_hours,
        48.0 + 72.0 * source_frame.mean_source_probability,
        marker="s",
        label="source probability (scaled for display)",
    )
    ax.set_xlabel("image age (hours)")
    ax.set_ylabel("hours")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(ANALYSIS_DIR / "source_timing_by_image_age.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    ax.plot(np.arange(1, len(singular_values) + 1), singular_values, marker=".")
    ax.set_yscale("log")
    ax.set_xlabel("representation component")
    ax.set_ylabel("singular value")
    ax.grid(alpha=0.3)
    fig.savefig(ANALYSIS_DIR / "cnn_representation_spectrum.png", dpi=160)
    plt.close(fig)


@torch.no_grad()
def main():
    model = load_best_model(
        model_class=SolarWindArrivalTCNV9,
        architecture_name="SolarWindArrivalTCNV9",
        file_stem=FILE_STEM,
    )
    val_chains = infer_temporal_chains(val_inputs, IMAGE_COLUMNS)
    dataset = ChainAwareSolarWindDataset(
        val_image_array,
        val_image_index,
        val_inputs,
        val_index,
        val_targets,
        temporal_chains=val_chains,
    )
    loader = make_chain_loader(dataset, training=False)

    predictions = []
    wind_predictions = []
    targets = []
    last_winds = []
    sample_ids = []
    arrival_weights = []
    transit_hours = []
    source_probability = []
    fusion_gates = []
    arrival_entropy = []
    image_tokens = []

    for batch_index, batch in enumerate(loader, start=1):
        if MAX_BATCHES > 0 and batch_index > MAX_BATCHES:
            break
        images = batch["images"].to(DEVICE, non_blocking=PIN_MEMORY)
        wind = batch["wind"].to(DEVICE, non_blocking=PIN_MEMORY)
        with torch.amp.autocast(AMP_DEVICE_TYPE, enabled=USE_AMP):
            output = model(
                images,
                wind,
                return_components=True,
                return_diagnostics=True,
            )
        prediction, wind_prediction, _, _, diagnostics = output
        predictions.append(prediction.float().cpu().numpy() * 1000.0)
        wind_predictions.append(wind_prediction.float().cpu().numpy() * 1000.0)
        targets.append(batch["target"].numpy() * 1000.0)
        last_winds.append(batch["wind"][:, -1].numpy() * 1000.0)
        sample_ids.extend(batch["sample_id"])
        arrival_weights.append(diagnostics["arrival_weights"].float().cpu().numpy())
        transit_hours.append(diagnostics["transit_hours"].float().cpu().numpy())
        source_probability.append(
            diagnostics["source_probability"].float().cpu().numpy()
        )
        fusion_gates.append(diagnostics["fusion_gate"].float().cpu().numpy())
        arrival_entropy.append(
            diagnostics["arrival_entropy"].float().cpu().numpy()
        )
        image_tokens.append(diagnostics["image_tokens"].float().cpu().numpy())
        print(f"diagnostic batch={batch_index}/{len(loader)}", flush=True)

    prediction = np.concatenate(predictions)
    wind_prediction = np.concatenate(wind_predictions)
    target = np.concatenate(targets)
    last_wind = np.concatenate(last_winds)
    weights = np.concatenate(arrival_weights)
    transit = np.concatenate(transit_hours)
    source = np.concatenate(source_probability)
    fusion_gate = np.concatenate(fusion_gates)
    entropy = np.concatenate(arrival_entropy)
    representations = np.concatenate(image_tokens)

    mean_weights = weights.mean(axis=0)
    observed_age = np.arange(19, -1, -1, dtype=np.float64) * 6.0
    expected_age = np.sum(mean_weights * observed_age[None, :], axis=1)
    horizon_frame = pd.DataFrame(
        {
            "horizon_hours": np.arange(1, 13) * 6,
            "expected_image_age_hours": expected_age,
            "normalized_gate_entropy": entropy.mean(axis=0),
            "maximum_gate_weight": weights.max(axis=-1).mean(axis=0),
            "mean_fusion_gate": fusion_gate.mean(axis=0),
            "rmse_km_s": np.sqrt(np.mean(np.square(prediction - target), axis=0)),
            "wind_only_rmse_km_s": np.sqrt(
                np.mean(np.square(wind_prediction - target), axis=0)
            ),
        }
    )
    horizon_frame.to_csv(ANALYSIS_DIR / "horizon_summary.csv", index=False)

    source_frame = pd.DataFrame(
        {
            "input_index": np.arange(20),
            "image_age_hours": observed_age,
            "mean_transit_hours": transit.mean(axis=0),
            "std_transit_hours": transit.std(axis=0),
            "mean_source_probability": source.mean(axis=0),
        }
    )
    source_frame.to_csv(ANALYSIS_DIR / "source_timing_summary.csv", index=False)

    gate_frame = pd.DataFrame(
        mean_weights,
        columns=[f"image_age_{int(age):03d}h" for age in observed_age],
    )
    gate_frame.insert(0, "horizon_hours", np.arange(1, 13) * 6)
    gate_frame.to_csv(ANALYSIS_DIR / "arrival_gate_mean.csv", index=False)

    delta = target - last_wind[:, None]
    event = np.max(np.abs(delta), axis=1) >= EVENT_THRESHOLD_KM_S
    per_sample = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "rmse_km_s": np.sqrt(np.mean(np.square(prediction - target), axis=1)),
            "wind_only_rmse_km_s": np.sqrt(
                np.mean(np.square(wind_prediction - target), axis=1)
            ),
            "is_event": event,
            "mean_transit_hours": transit.mean(axis=1),
            "mean_source_probability": source.mean(axis=1),
            "mean_arrival_entropy": entropy.mean(axis=1),
            "mean_fusion_gate": fusion_gate.mean(axis=1),
        }
    )
    per_sample.to_csv(ANALYSIS_DIR / "per_sample.csv", index=False)

    representation_summary, singular_values = representation_statistics(
        representations
    )
    full_rmse = rmse(prediction - target)
    wind_rmse = rmse(wind_prediction - target)
    summary = {
        "sample_count": int(len(target)),
        "overall_rmse_km_s": full_rmse,
        "wind_only_rmse_km_s": wind_rmse,
        "image_gain_km_s": wind_rmse - full_rmse,
        "event_fraction": float(event.mean()),
        "event_rmse_km_s": rmse((prediction - target)[event]) if event.any() else None,
        "quiet_rmse_km_s": rmse((prediction - target)[~event]) if (~event).any() else None,
        "mean_transit_hours": float(transit.mean()),
        "std_transit_hours": float(transit.std()),
        "mean_source_probability": float(source.mean()),
        "std_source_probability": float(source.std()),
        "mean_arrival_entropy": float(entropy.mean()),
        "mean_fusion_gate": float(fusion_gate.mean()),
        "arrival_prior_strength": float(
            F.softplus(model.arrival_prior_raw).detach().cpu()
        ),
        "arrival_sigma_hours": float(
            model.log_arrival_sigma.exp().detach().cpu()
        ),
        "cnn_representation": representation_summary,
    }
    (ANALYSIS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    save_gate_plots(mean_weights, horizon_frame, source_frame, singular_values)

    print(json.dumps(summary, indent=2))
    print(f"saved diagnostics: {ANALYSIS_DIR.resolve()}")


if __name__ == "__main__":
    main()
