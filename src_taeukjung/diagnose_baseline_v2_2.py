import json
import os

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
from diagnose_baseline_v2_1 import save_preprocessing_figure
from inference_solar_physics_v5 import load_best_model
from model_baseline_v2_2 import SolarWindBaselineSpatialTransformerV22


FILE_STEM = "baseline_v2_2"
ARCHITECTURE_NAME = "SolarWindBaselineSpatialTransformerV22"


def representation_metrics(memory):
    flattened = memory.reshape(-1, memory.shape[-1]).astype(np.float64)
    centered = flattened - flattened.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = singular_values**2
    probability = energy / max(float(energy.sum()), 1e-12)
    effective_rank = float(np.exp(-np.sum(probability * np.log(probability + 1e-12))))
    feature_std = flattened.std(axis=0)

    normalized = F.normalize(torch.from_numpy(memory).float(), dim=-1)
    temporal_cosine = float(
        (normalized[:, 1:] * normalized[:, :-1]).sum(-1).mean()
    )
    horizontal_cosine = float(
        (normalized[:, :, :, 1:] * normalized[:, :, :, :-1]).sum(-1).mean()
    )
    vertical_cosine = float(
        (normalized[:, :, 1:] * normalized[:, :, :-1]).sum(-1).mean()
    )
    return {
        "effective_rank": effective_rank,
        "feature_dimension": int(memory.shape[-1]),
        "token_count_per_sample": int(np.prod(memory.shape[1:-1])),
        "median_feature_std": float(np.median(feature_std)),
        "minimum_feature_std": float(np.min(feature_std)),
        "collapsed_feature_fraction": float(np.mean(feature_std < 1e-3)),
        "adjacent_time_cosine": temporal_cosine,
        "adjacent_horizontal_cosine": horizontal_cosine,
        "adjacent_vertical_cosine": vertical_cosine,
        "singular_values": singular_values,
    }


def save_temporal_attention(mean_attention, output_path):
    figure, axis = plt.subplots(figsize=(10, 5))
    image = axis.imshow(mean_attention, aspect="auto", cmap="viridis")
    axis.set_xlabel("Image age (hours)")
    axis.set_ylabel("Forecast horizon (hours)")
    axis.set_xticks(np.arange(20))
    axis.set_xticklabels(np.arange(19, -1, -1) * 6, rotation=45)
    axis.set_yticks(np.arange(12))
    axis.set_yticklabels(np.arange(1, 13) * 6)
    figure.colorbar(image, ax=axis, label="Mean temporal attention")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_spatial_attention(mean_spatial, height, width, output_path):
    figure, axes = plt.subplots(
        3, 4, figsize=(11, 7.5), constrained_layout=True
    )
    lower = float(mean_spatial.min())
    upper = float(mean_spatial.max())
    for horizon_index, axis in enumerate(axes.flat):
        image = axis.imshow(
            mean_spatial[horizon_index].reshape(height, width),
            cmap="magma",
            vmin=lower,
            vmax=upper,
        )
        axis.set_title(f"+{(horizon_index + 1) * 6} h")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        label="Mean spatial attention",
        shrink=0.85,
        pad=0.02,
    )
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_selected_token_attention(mean_tokens, output_path):
    selected = (0, 5, 11)
    figure, axes = plt.subplots(
        1, 3, figsize=(14, 4.5), sharey=True, constrained_layout=True
    )
    lower = float(mean_tokens[list(selected)].min())
    upper = float(mean_tokens[list(selected)].max())
    for axis, horizon_index in zip(axes, selected):
        image = axis.imshow(
            mean_tokens[horizon_index].T,
            aspect="auto",
            cmap="viridis",
            vmin=lower,
            vmax=upper,
        )
        axis.set_title(f"Forecast +{(horizon_index + 1) * 6} h")
        axis.set_xlabel("Image age (hours)")
        axis.set_xticks(np.arange(0, 20, 3))
        axis.set_xticklabels(np.arange(114, -1, -18), rotation=45)
    axes[0].set_ylabel("Spatial cell index")
    figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        label="Token attention",
        shrink=0.85,
        pad=0.02,
    )
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


@torch.no_grad()
def main(
    model_class=SolarWindBaselineSpatialTransformerV22,
    architecture_name=ARCHITECTURE_NAME,
    file_stem=FILE_STEM,
):
    model = load_best_model(
        model_class,
        architecture_name,
        file_stem,
        grid_label="spatial_grid",
    )
    chains = infer_temporal_chains(val_inputs, IMAGE_COLUMNS)
    sample_limit = int(os.getenv("DIAGNOSTIC_SAMPLES", "512"))
    indexes = val_index[: min(sample_limit, len(val_index))]
    dataset = ChainAwareSolarWindDataset(
        val_image_array,
        val_image_index,
        val_inputs,
        indexes,
        val_targets,
        temporal_chains=chains,
    )
    loader = make_chain_loader(dataset, training=False)

    predictions = []
    targets = []
    temporal_attention = []
    spatial_attention = []
    token_attention = []
    expected_age = []
    temporal_entropy = []
    spatial_entropy = []
    gates = []
    image_scales = []
    transit = []
    memories = []
    first_preprocess = None
    for batch_index, batch in enumerate(loader, start=1):
        images = batch["images"].to(DEVICE, non_blocking=PIN_MEMORY)
        wind = batch["wind"].to(DEVICE, non_blocking=PIN_MEMORY)
        with torch.amp.autocast(AMP_DEVICE_TYPE, enabled=USE_AMP):
            prediction, diagnostics = model(
                images, wind, return_diagnostics=True
            )
        predictions.append(prediction.float().cpu().numpy() * 1000.0)
        targets.append(batch["target"].numpy() * 1000.0)
        temporal_attention.append(
            diagnostics["attention_weights"].float().cpu().numpy()
        )
        spatial_attention.append(
            diagnostics["attention_spatial_weights"].float().cpu().numpy()
        )
        token_attention.append(
            diagnostics["attention_token_weights"].float().cpu().numpy()
        )
        expected_age.append(
            diagnostics["attention_expected_age_hours"].float().cpu().numpy()
        )
        temporal_entropy.append(
            diagnostics["attention_entropy"].float().cpu().numpy()
        )
        spatial_entropy.append(
            diagnostics["attention_spatial_entropy"].float().cpu().numpy()
        )
        gates.append(diagnostics["image_gate"].float().cpu().numpy())
        if "image_scale_fraction" in diagnostics:
            image_scales.append(
                diagnostics["image_scale_fraction"].float().cpu().numpy()
            )
        transit.append(
            diagnostics["nominal_transit_hours"].float().cpu().numpy()
        )
        memories.append(
            diagnostics["image_memory"].float().cpu().numpy().reshape(
                images.shape[0],
                20,
                model.spatial_height,
                model.spatial_width,
                model.d_model,
            )
        )
        if first_preprocess is None:
            first_preprocess = (
                images[0].float().cpu().numpy(),
                diagnostics["masked_images"][0].float().cpu().numpy(),
                diagnostics["delta_images"][0].float().cpu().numpy(),
            )
        print(f"diagnostic batch={batch_index}/{len(loader)}", flush=True)

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    temporal_attention = np.concatenate(temporal_attention)
    spatial_attention = np.concatenate(spatial_attention)
    token_attention = np.concatenate(token_attention)
    expected_age = np.concatenate(expected_age)
    temporal_entropy = np.concatenate(temporal_entropy)
    spatial_entropy = np.concatenate(spatial_entropy)
    gates = np.concatenate(gates)
    if image_scales:
        image_scales = np.concatenate(image_scales)
    transit = np.concatenate(transit)
    memories = np.concatenate(memories)

    mean_temporal = temporal_attention.mean(axis=0)
    mean_spatial = spatial_attention.mean(axis=0)
    mean_tokens = token_attention.mean(axis=0)
    diagnostic_dir = OUTPUT_DIR / f"{file_stem}_diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    save_preprocessing_figure(
        *first_preprocess, diagnostic_dir / "preprocessing_delta.png"
    )
    save_temporal_attention(
        mean_temporal, diagnostic_dir / "temporal_attention_mean.png"
    )
    save_spatial_attention(
        mean_spatial,
        model.spatial_height,
        model.spatial_width,
        diagnostic_dir / "spatial_attention_by_horizon.png",
    )
    save_selected_token_attention(
        mean_tokens, diagnostic_dir / "spatiotemporal_attention_selected.png"
    )

    representation = representation_metrics(memories)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.semilogy(representation.pop("singular_values"))
    axis.set_xlabel("Component")
    axis.set_ylabel("Singular value")
    axis.grid(True)
    figure.tight_layout()
    figure.savefig(diagnostic_dir / "token_spectrum.png", dpi=160)
    plt.close(figure)

    pd.DataFrame(
        mean_temporal,
        index=[f"horizon_{hour:02d}h" for hour in range(6, 73, 6)],
        columns=[f"age_{hour:03d}h" for hour in range(114, -1, -6)],
    ).to_csv(diagnostic_dir / "temporal_attention_mean.csv")
    pd.DataFrame(
        mean_spatial,
        index=[f"horizon_{hour:02d}h" for hour in range(6, 73, 6)],
        columns=[f"cell_{index:02d}" for index in range(model.memory_spatial_tokens)],
    ).to_csv(diagnostic_dir / "spatial_attention_mean.csv")
    horizon_summary = {
        "horizon_hours": np.arange(1, 13) * 6,
        "expected_image_age_hours": expected_age.mean(axis=0),
        "temporal_attention_entropy": temporal_entropy.mean(axis=0),
        "spatial_attention_entropy": spatial_entropy.mean(axis=0),
        "image_gate": gates.mean(axis=0),
    }
    if hasattr(model, "fixed_lag_hours"):
        horizon_summary["fixed_source_age_hours"] = (
            model.fixed_lag_hours - np.arange(1, 13) * 6
        )
    if len(image_scales):
        horizon_summary["mean_image_scale_percent"] = (
            image_scales.mean(axis=0) * 100.0
        )
        horizon_summary["mean_absolute_image_scale_percent"] = (
            np.abs(image_scales).mean(axis=0) * 100.0
        )
    pd.DataFrame(horizon_summary).to_csv(
        diagnostic_dir / "horizon_summary.csv", index=False
    )

    summary = {
        "sample_count": int(len(predictions)),
        "rmse_km_s": float(np.sqrt(np.mean((predictions - targets) ** 2))),
        "mean_nominal_transit_hours": float(transit.mean()),
        "mean_temporal_attention_entropy": float(temporal_entropy.mean()),
        "mean_spatial_attention_entropy": float(spatial_entropy.mean()),
        "mean_image_gate": float(gates.mean()),
        "transformer_value_count": model.transformer_value_count(),
        "encoder_attention_score_count_per_head": model.encoder_attention_score_count(),
        **representation,
    }
    if hasattr(model, "fixed_lag_hours"):
        summary.update(
            {
                "fixed_lag_hours": model.fixed_lag_hours,
                "fixed_lag_sigma_hours": model.fixed_lag_sigma_hours,
                "fixed_lag_window_hours": model.fixed_lag_window_hours,
            }
        )
    if len(image_scales):
        summary["mean_absolute_image_scale_percent"] = float(
            np.abs(image_scales).mean() * 100.0
        )
    (diagnostic_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"saved diagnostics: {diagnostic_dir.resolve()}")


if __name__ == "__main__":
    main()
