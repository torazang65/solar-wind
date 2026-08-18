import json
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from config import *

IMAGE_COLUMNS = [f"image_{index:02d}" for index in range(20)]
WIND_COLUMNS = [f"wind_{index:02d}" for index in range(20)]
TARGET_COLUMNS = [f"target_{index:02d}" for index in range(12)]

# Load DataFrames
info = json.loads((DATA_ROOT / "dataset_info.json").read_text(encoding="utf-8"))
train_inputs = pd.read_csv(DATA_ROOT / "train/inputs.csv")
train_targets_frame = pd.read_csv(DATA_ROOT / "train/targets.csv")
val_inputs = pd.read_csv(DATA_ROOT / "validation/inputs.csv")
val_targets_frame = pd.read_csv(DATA_ROOT / "validation/targets.csv")
test_inputs = pd.read_csv(DATA_ROOT / "test/inputs.csv")
test_ids = pd.read_csv(DATA_ROOT / "test/test_ids.csv")

train_targets = train_targets_frame[TARGET_COLUMNS].to_numpy(np.float32)
val_targets = val_targets_frame[TARGET_COLUMNS].to_numpy(np.float32)
train_index = np.arange(len(train_inputs), dtype=np.int64)
val_index = np.arange(len(val_inputs), dtype=np.int64)
test_index = np.arange(len(test_inputs), dtype=np.int64)

def prepare_image_memmap(split, inputs):
    image_root = DATA_ROOT / split
    cache_root = DATA_ROOT / "resized_cache" / f"{IMAGE_SIZE}px"
    cache_root.mkdir(parents=True, exist_ok=True)
    array_path = cache_root / f"{split}_images.npy"
    metadata_path = cache_root / f"{split}_metadata.json"
    
    filenames = sorted(pd.unique(inputs[IMAGE_COLUMNS].to_numpy().ravel()).tolist())
    expected = {
        "image_size": IMAGE_SIZE,
        "channels": list(CHANNELS),
        "filenames": filenames,
    }
    valid = False
    if array_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cached = np.load(array_path, mmap_mode="r")
            valid = metadata == expected and cached.shape == (
                len(filenames), len(CHANNELS), IMAGE_SIZE, IMAGE_SIZE
            ) and cached.dtype == np.uint8
        except (OSError, ValueError, json.JSONDecodeError):
            valid = False
            
    if not valid:
        array_temp = array_path.with_name(array_path.name + f".partial.{os.getpid()}")
        metadata_temp = metadata_path.with_name(metadata_path.name + f".partial.{os.getpid()}")
        resized_images = np.lib.format.open_memmap(
            array_temp, mode="w+", dtype=np.uint8,
            shape=(len(filenames), len(CHANNELS), IMAGE_SIZE, IMAGE_SIZE),
        )
        resampling = Image.Resampling.BILINEAR
        for index, filename in enumerate(filenames):
            for channel_index, channel in enumerate(CHANNELS):
                with Image.open(image_root / channel / filename) as image:
                    resized_images[index, channel_index] = np.asarray(
                        image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), resampling), 
                        dtype=np.uint8,
                    )
            if (index + 1) % 2000 == 0 or index + 1 == len(filenames):
                print(f"{split} resize: {index + 1}/{len(filenames)}", flush=True)
        resized_images.flush()
        del resized_images
        metadata_temp.write_text(json.dumps(expected, ensure_ascii=False) + "\n", encoding="utf-8")
        array_temp.replace(array_path)
        metadata_temp.replace(metadata_path)
        print(f"created resized cache: {array_path.resolve()}")
    else:
        print(f"reusing resized cache: {array_path.resolve()}")
        
    image_array = np.load(array_path, mmap_mode="r")
    image_index = {filename: index for index, filename in enumerate(filenames)}
    return image_array, image_index

train_image_array, train_image_index = prepare_image_memmap("train", train_inputs)
val_image_array, val_image_index = prepare_image_memmap("validation", val_inputs)
test_image_array, test_image_index = prepare_image_memmap("test", test_inputs)

class SolarWindDataset(Dataset):
    def __init__(self, image_array, image_index, inputs, indexes, targets=None):
        self.image_array = image_array
        self.image_indexes = np.asarray([
            [image_index[filename] for filename in row]
            for row in inputs[IMAGE_COLUMNS].itertuples(index=False, name=None)
        ], dtype=np.int32)
        self.wind = inputs[WIND_COLUMNS].to_numpy(np.float32) / 1000.0
        self.sample_ids = inputs.sample_id.to_numpy()
        self.indexes = np.asarray(indexes, dtype=np.int64)
        self.targets = targets

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, item):
        row_index = int(self.indexes[item])
        images = np.asarray(self.image_array[self.image_indexes[row_index]], dtype=np.float32) / 255.0
        result = {
            "images": torch.from_numpy(images),
            "wind": torch.from_numpy(self.wind[row_index]),
            "sample_id": self.sample_ids[row_index],
        }
        if self.targets is not None:
            result["target"] = torch.from_numpy(np.asarray(self.targets[row_index], dtype=np.float32) / 1000.0)
        return result

def seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def make_loader(dataset, shuffle):
    options = dict(
        dataset=dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False,
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(SEED),
    )
    if NUM_WORKERS > 0:
        options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**options)

train_dataset = SolarWindDataset(train_image_array, train_image_index, train_inputs, train_index, train_targets)
val_dataset = SolarWindDataset(val_image_array, val_image_index, val_inputs, val_index, val_targets)

train_loader = make_loader(train_dataset, shuffle=True)
val_loader = make_loader(val_dataset, shuffle=False)