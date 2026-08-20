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
    TARGET_COLUMNS,
    val_image_array,
    val_image_index,
    val_index,
    val_inputs,
    val_targets,
)
from inference_solar_physics_v5 import load_best_model
from model_baseline_v2_1 import SolarWindBaselineTransformerV21


FILE_STEM = "baseline_v2_1"
ARCHITECTURE_NAME = "SolarWindBaselineTransformerV21"


def save_preprocessing_figure(raw, masked, delta, output_path):
    figure, axes = plt.subplots(2, 4, figsize=(13, 6))
    columns = [
        (raw[-2], "input t-1"),
        (raw[-1], "input t"),
        (masked[-1], "masked t"),
        (delta[-1], "scaled delta t"),
    ]
    for channel in range(2):
        for column, (values, title) in enumerate(columns):
            image = values[channel]
            if column == 3:
                limit = max(float(np.max(np.abs(image))), 1e-3)
                axes[channel, column].imshow(
                    image, cmap="coolwarm", vmin=-limit, vmax=limit
                )
            else:
                axes[channel, column].imshow(image, cmap="gray", vmin=0, vmax=1)
            axes[channel, column].set_title(
                f"{('193A', '211A')[channel]} {title}"
            )
            axes[channel, column].axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def representation_metrics(memory):
    flattened = memory.reshape(-1, memory.shape[-1]).astype(np.float64)
    centered = flattened - flattened.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = singular_values**2
    probability = energy / max(float(energy.sum()), 1e-12)
    effective_rank = float(np.exp(-np.sum(probability * np.log(probability + 1e-12))))
    feature_std = flattened.std(axis=0)
    normalized = F.normalize(torch.from_numpy(memory).float(), dim=-1)
    adjacent = float((normalized[:, 1:] * normalized[:, :-1]).sum(-1).mean())
    return {
        "effective_rank": effective_rank,
        "feature_dimension": int(memory.shape[-1]),
        "median_feature_std": float(np.median(feature_std)),
        "minimum_feature_std": float(np.min(feature_std)),
        "collapsed_feature_fraction": float(np.mean(feature_std < 1e-3)),
        "adjacent_time_cosine": adjacent,
        "singular_values": singular_values,
    }


@torch.no_grad()
def main():
    model = load_best_model(
        SolarWindBaselineTransformerV21,
        ARCHITECTURE_NAME,
        FILE_STEM,
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
    attention = []
    expected_age = []
    entropy = []
    gates = []
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
        attention.append(diagnostics["attention_weights"].float().cpu().numpy())
        expected_age.append(
            diagnostics["attention_expected_age_hours"].float().cpu().numpy()
        )
        entropy.append(diagnostics["attention_entropy"].float().cpu().numpy())
        gates.append(diagnostics["image_gate"].float().cpu().numpy())
        transit.append(
            diagnostics["nominal_transit_hours"].float().cpu().numpy()
        )
        memories.append(diagnostics["image_memory"].float().cpu().numpy())
        if first_preprocess is None:
            first_preprocess = (
                images[0].float().cpu().numpy(),
                diagnostics["masked_images"][0].float().cpu().numpy(),
                diagnostics["delta_images"][0].float().cpu().numpy(),
            )
        print(f"diagnostic batch={batch_index}/{len(loader)}", flush=True)

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    attention = np.concatenate(attention)
    expected_age = np.concatenate(expected_age)
    entropy = np.concatenate(entropy)
    gates = np.concatenate(gates)
    transit = np.concatenate(transit)
    memories = np.concatenate(memories)
    mean_attention = attention.mean(axis=0)

    diagnostic_dir = OUTPUT_DIR / f"{FILE_STEM}_diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    save_preprocessing_figure(
        *first_preprocess, diagnostic_dir / "preprocessing_delta.png"
    )

    figure, axis = plt.subplots(figsize=(10, 5))
    image = axis.imshow(mean_attention, aspect="auto", cmap="viridis")
    axis.set_xlabel("Image age (hours)")
    axis.set_ylabel("Forecast horizon (hours)")
    axis.set_xticks(np.arange(20))
    axis.set_xticklabels(np.arange(19, -1, -1) * 6, rotation=45)
    axis.set_yticks(np.arange(12))
    axis.set_yticklabels(np.arange(1, 13) * 6)
    figure.colorbar(image, ax=axis, label="Mean attention")
    figure.tight_layout()
    figure.savefig(diagnostic_dir / "attention_mean.png", dpi=160)
    plt.close(figure)

    representation = representation_metrics(memories)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.semilogy(representation.pop("singular_values"))
    axis.set_xlabel("Component")
    axis.set_ylabel("Singular value")
    axis.grid(True)
    figure.tight_layout()
    figure.savefig(diagnostic_dir / "cnn_token_spectrum.png", dpi=160)
    plt.close(figure)

    pd.DataFrame(
        mean_attention,
        index=[f"horizon_{hour:02d}h" for hour in range(6, 73, 6)],
        columns=[f"age_{hour:03d}h" for hour in range(114, -1, -6)],
    ).to_csv(diagnostic_dir / "attention_mean.csv")
    pd.DataFrame(
        {
            "horizon_hours": np.arange(1, 13) * 6,
            "expected_image_age_hours": expected_age.mean(axis=0),
            "attention_entropy": entropy.mean(axis=0),
            "image_gate": gates.mean(axis=0),
        }
    ).to_csv(diagnostic_dir / "horizon_summary.csv", index=False)

    summary = {
        "sample_count": int(len(predictions)),
        "rmse_km_s": float(np.sqrt(np.mean((predictions - targets) ** 2))),
        "mean_nominal_transit_hours": float(transit.mean()),
        "mean_attention_entropy": float(entropy.mean()),
        "mean_image_gate": float(gates.mean()),
        **representation,
    }
    (diagnostic_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"saved diagnostics: {diagnostic_dir.resolve()}")


if __name__ == "__main__":
    main()
