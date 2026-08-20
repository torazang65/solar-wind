import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_ROOT = REPO_ROOT / "src_taeukjung"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


IMAGE_COLUMNS = [f"image_{index:02d}" for index in range(20)]
WIND_COLUMNS = [f"wind_{index:02d}" for index in range(20)]
TARGET_COLUMNS = [f"target_{index:02d}" for index in range(12)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 4 fixed ablations (baseline, mask, soft cubic, mask+soft) for 20 epochs."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "public_dataset/competition_dataset_6h",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs/ablation_64_20epoch",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--radius-fraction", type=float, default=0.49)
    parser.add_argument("--soft-cubic-strength", type=float, default=0.25)
    parser.add_argument("--train-limit", type=int, default=0, help="0 means full train split.")
    parser.add_argument("--val-limit", type=int, default=0, help="0 means full validation split.")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), True
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), False
    return torch.device("cpu"), False


def choose_subset(inputs, targets, limit, seed):
    if limit <= 0 or limit >= len(inputs):
        return inputs.reset_index(drop=True), targets.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(inputs), size=limit, replace=False))
    sampled_inputs = inputs.iloc[selected].reset_index(drop=True)
    sampled_targets = targets.set_index("sample_id").loc[sampled_inputs.sample_id].reset_index()
    return sampled_inputs, sampled_targets


def normalize_uint8(images, mode, strength):
    x = images.astype(np.float32) / 255.0
    if mode == "linear":
        return x
    pure = 0.5 * ((2.0 * x - 1.0) ** 3 + 1.0)
    if mode == "cubic":
        return pure
    if mode == "soft_cubic":
        return (1.0 - strength) * x + strength * pure
    raise ValueError(f"unknown IMAGE_NORM mode: {mode}")


def prepare_image_cache(split, inputs, data_root, cache_root, image_size):
    split_root = data_root / split
    cache_root.mkdir(parents=True, exist_ok=True)
    filenames = sorted(pd.unique(inputs[IMAGE_COLUMNS].to_numpy().ravel()).tolist())
    array_path = cache_root / f"{split}_{image_size}px_images.npy"
    metadata_path = cache_root / f"{split}_{image_size}px_metadata.json"

    expected = {
        "split": split,
        "image_size": image_size,
        "channels": ["193", "211"],
        "filenames": filenames,
        "preprocess": {
            "color_mode": "L",
            "resize_resampling": "BILINEAR",
            "image_norm_applied_to_cache": False,
        },
    }

    valid = False
    if array_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cached = np.load(array_path, mmap_mode="r")
            valid = (
                metadata == expected
                and cached.shape == (len(filenames), 2, image_size, image_size)
                and cached.dtype == np.uint8
            )
        except (OSError, ValueError, json.JSONDecodeError):
            valid = False

    if not valid:
        temp_array = array_path.with_name(array_path.name + f".partial.{os.getpid()}")
        temp_metadata = metadata_path.with_name(metadata_path.name + f".partial.{os.getpid()}")
        resized_images = np.lib.format.open_memmap(
            temp_array,
            mode="w+",
            dtype=np.uint8,
            shape=(len(filenames), 2, image_size, image_size),
        )
        resampling = Image.Resampling.BILINEAR
        for index, filename in enumerate(filenames):
            for channel_index, channel in enumerate(["193", "211"]):
                with Image.open(split_root / channel / filename) as image:
                    gray = image.convert("L")
                    if gray.size != (image_size, image_size):
                        gray = gray.resize((image_size, image_size), resampling)
                    resized_images[index, channel_index] = np.asarray(gray, dtype=np.uint8)
            if (index + 1) % 500 == 0 or index + 1 == len(filenames):
                print(f"{split} cache: {index + 1}/{len(filenames)}", flush=True)
        resized_images.flush()
        del resized_images
        temp_metadata.write_text(json.dumps(expected, ensure_ascii=False) + "\n", encoding="utf-8")
        temp_array.replace(array_path)
        temp_metadata.replace(metadata_path)
        print(f"created resized cache: {array_path.resolve()}")
    else:
        print(f"reusing resized cache: {array_path.resolve()}")

    image_array = np.load(array_path, mmap_mode="r")
    image_index = {filename: index for index, filename in enumerate(filenames)}
    return image_array, image_index


class SolarWindSubsetDataset(Dataset):
    def __init__(
        self,
        image_array,
        image_index,
        inputs,
        targets,
        norm_mode,
        soft_strength,
    ):
        self.image_array = image_array
        self.image_indexes = np.asarray(
            [[image_index[filename] for filename in row] for row in inputs[IMAGE_COLUMNS].itertuples(index=False, name=None)],
            dtype=np.int32,
        )
        self.wind = inputs[WIND_COLUMNS].to_numpy(np.float32) / 1000.0
        self.sample_ids = inputs.sample_id.to_numpy()
        self.targets = targets[TARGET_COLUMNS].to_numpy(np.float32) / 1000.0
        self.norm_mode = norm_mode
        self.soft_strength = soft_strength

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, item):
        images = normalize_uint8(self.image_array[self.image_indexes[item]], self.norm_mode, self.soft_strength)
        return {
            "images": torch.from_numpy(images),
            "wind": torch.from_numpy(self.wind[item]),
            "target": torch.from_numpy(self.targets[item]),
            "sample_id": self.sample_ids[item],
        }


def make_loaders(split_data, norm_mode, batch_size, num_workers, seed, soft_strength, device_type):
    train_dataset = SolarWindSubsetDataset(
        split_data["train_image_array"],
        split_data["train_image_index"],
        split_data["train_inputs"],
        split_data["train_targets"],
        norm_mode,
        soft_strength,
    )
    val_dataset = SolarWindSubsetDataset(
        split_data["val_image_array"],
        split_data["val_image_index"],
        split_data["val_inputs"],
        split_data["val_targets"],
        norm_mode,
        soft_strength,
    )

    generator = torch.Generator().manual_seed(seed)
    pin_memory = device_type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def run_epoch(model, loader, device, optimizer, scaler, use_amp):
    training = optimizer is not None
    model.train(training)
    squared_sum = 0.0
    mae_sum = 0.0
    value_count = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=device.type == "cuda")
        wind = batch["wind"].to(device, non_blocking=device.type == "cuda")
        target = batch["target"].to(device, non_blocking=device.type == "cuda")

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device.type, enabled=use_amp):
                prediction = model(images, wind)
                loss = torch.sqrt(F.mse_loss(prediction, target) + 1e-8)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        diff = (prediction.detach() - target) * 1000.0
        squared_sum += float(torch.sum(diff ** 2).cpu())
        mae_sum += float(torch.sum(torch.abs(diff)).cpu())
        value_count += diff.numel()

    return {
        "rmse_km_s": math.sqrt(squared_sum / value_count),
        "mae_km_s": mae_sum / value_count,
    }


def run_experiment(cfg, split_data, args, device, output_dir):
    from model import SolarWindBaseline

    print(f"\n[experiment] {cfg['name']} mask={cfg['mask']} norm={cfg['norm']}")

    train_loader, val_loader = make_loaders(
        split_data=split_data,
        norm_mode=cfg["norm"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        soft_strength=args.soft_cubic_strength,
        device_type=device.type,
    )

    model = SolarWindBaseline(
        image_size=args.image_size,
        apply_solar_disk_mask=cfg["mask"],
        solar_disk_radius_fraction=args.radius_fraction,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.25,
        patience=1,
        min_lr=1e-6,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best = {"rmse": float("inf"), "epoch": 0, "state": None}
    history_rows = []
    exp_dir = output_dir / cfg["name"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, device, optimizer, scaler, use_amp)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device, optimizer=None, scaler=None, use_amp=use_amp)

        elapsed = time.perf_counter() - started
        scheduler.step(val_metrics["rmse_km_s"])
        lr = optimizer.param_groups[0]["lr"]

        history_rows.append(
            {
                "epoch": epoch,
                "train_rmse_km_s": train_metrics["rmse_km_s"],
                "train_mae_km_s": train_metrics["mae_km_s"],
                "val_rmse_km_s": val_metrics["rmse_km_s"],
                "val_mae_km_s": val_metrics["mae_km_s"],
                "learning_rate": lr,
                "seconds": elapsed,
            }
        )

        if val_metrics["rmse_km_s"] < best["rmse"]:
            best["rmse"] = val_metrics["rmse_km_s"]
            best["epoch"] = epoch
            best["state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(
            f"{cfg['name']} epoch={epoch:02d} "
            f"train_rmse={train_metrics['rmse_km_s']:.3f} "
            f"val_rmse={val_metrics['rmse_km_s']:.3f} "
            f"lr={lr:.2e} sec={elapsed:.1f}",
            flush=True,
        )

    history = pd.DataFrame(history_rows)
    history.to_csv(exp_dir / "history.csv", index=False)
    torch.save(
        {
            "model_state_dict": best["state"],
            "best_epoch": best["epoch"],
            "best_val_rmse_km_s": best["rmse"],
            "name": cfg["name"],
            "mask": cfg["mask"],
            "norm": cfg["norm"],
            "image_size": args.image_size,
            "radius_fraction": args.radius_fraction,
            "soft_cubic_strength": args.soft_cubic_strength,
        },
        exp_dir / "best_model.pth",
    )

    best_row = history.loc[history["val_rmse_km_s"].idxmin()]
    return {
        "name": cfg["name"],
        "mask": cfg["mask"],
        "norm": cfg["norm"],
        "best_epoch": int(best["epoch"]),
        "train_rmse_km_s": float(best_row["train_rmse_km_s"]),
        "val_rmse_km_s": float(best["rmse"]),
        "val_mae_km_s": float(best_row["val_mae_km_s"]),
    }


def main():
    args = parse_args()
    args.image_size = int(args.image_size)
    args.epochs = int(args.epochs)
    seed_everything(args.seed)

    run_name = args.run_name or time.strftime("ablation_4way_%Y%m%d_%H%M%S")
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    args.output_root.mkdir(parents=True, exist_ok=True)

    train_inputs_all = pd.read_csv(args.data_root / "train/inputs.csv")
    train_targets_all = pd.read_csv(args.data_root / "train/targets.csv")
    val_inputs_all = pd.read_csv(args.data_root / "validation/inputs.csv")
    val_targets_all = pd.read_csv(args.data_root / "validation/targets.csv")

    train_inputs, train_targets = choose_subset(
        train_inputs_all, train_targets_all, args.train_limit, seed=args.seed
    )
    val_inputs, val_targets = choose_subset(
        val_inputs_all, val_targets_all, args.val_limit, seed=args.seed + 1
    )

    train_inputs.to_csv(output_dir / "train_subset_inputs.csv", index=False)
    train_targets.to_csv(output_dir / "train_subset_targets.csv", index=False)
    val_inputs.to_csv(output_dir / "val_subset_inputs.csv", index=False)
    val_targets.to_csv(output_dir / "val_subset_targets.csv", index=False)

    train_image_array, train_image_index = prepare_image_cache(
        "train", train_inputs, args.data_root, cache_dir, args.image_size
    )
    val_image_array, val_image_index = prepare_image_cache(
        "validation", val_inputs, args.data_root, cache_dir, args.image_size
    )

    split_data = {
        "train_inputs": train_inputs,
        "train_targets": train_targets,
        "val_inputs": val_inputs,
        "val_targets": val_targets,
        "train_image_array": train_image_array,
        "train_image_index": train_image_index,
        "val_image_array": val_image_array,
        "val_image_index": val_image_index,
    }

    device, _ = get_device()
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    elif device.type == "mps":
        print("INFO: using Apple MPS.")

    experiments = [
        {"name": "baseline", "mask": False, "norm": "linear"},
        {"name": "mask_only", "mask": True, "norm": "linear"},
        {"name": "soft025_only", "mask": False, "norm": "soft_cubic"},
        {"name": "mask_soft025", "mask": True, "norm": "soft_cubic"},
    ]

    metadata = {
        "run_name": run_name,
        "data_root": str(args.data_root),
        "device": str(device),
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "lr": args.lr,
        "radius_fraction": args.radius_fraction,
        "soft_cubic_strength": args.soft_cubic_strength,
        "train_samples": int(len(train_inputs)),
        "val_samples": int(len(val_inputs)),
        "experiments": experiments,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_rows = []
    for cfg in experiments:
        row = run_experiment(cfg, split_data, args, device, output_dir)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    print(f"\n[summary]")
    print(summary.sort_values("val_rmse_km_s").to_string(index=False))
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
