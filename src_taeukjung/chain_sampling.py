from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from config import BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, SEED


@dataclass(frozen=True)
class TemporalChains:
    chain_ids: np.ndarray
    positions: np.ndarray
    chains: tuple

    @property
    def count(self):
        return len(self.chains)

    @property
    def lengths(self):
        return np.asarray([len(chain) for chain in self.chains], dtype=np.int64)


def infer_temporal_chains(inputs, image_columns):
    """Recover shuffled one-step sliding-window chains from image filenames."""
    sequences = inputs[image_columns].astype(str).to_numpy()
    if sequences.shape[1] < 2:
        raise ValueError("at least two image columns are required")

    prefix_to_row = {}
    for row_index, sequence in enumerate(sequences):
        prefix = tuple(sequence[:-1])
        if prefix in prefix_to_row:
            raise ValueError("image-window prefixes are not unique")
        prefix_to_row[prefix] = row_index

    successor = np.full(len(sequences), -1, dtype=np.int64)
    predecessor = np.full(len(sequences), -1, dtype=np.int64)
    for row_index, sequence in enumerate(sequences):
        next_row = prefix_to_row.get(tuple(sequence[1:]))
        if next_row is None:
            continue
        if predecessor[next_row] >= 0:
            raise ValueError("a temporal row has multiple predecessors")
        successor[row_index] = next_row
        predecessor[next_row] = row_index

    chain_ids = np.full(len(sequences), -1, dtype=np.int64)
    positions = np.full(len(sequences), -1, dtype=np.int64)
    chains = []

    starts = np.flatnonzero(predecessor < 0)
    starts = sorted(starts, key=lambda index: tuple(sequences[index]))
    for start in starts:
        chain = []
        row_index = int(start)
        while row_index >= 0:
            if chain_ids[row_index] >= 0:
                raise ValueError("cycle or merged temporal chain detected")
            chain_ids[row_index] = len(chains)
            positions[row_index] = len(chain)
            chain.append(row_index)
            row_index = int(successor[row_index])
        chains.append(tuple(chain))

    if np.any(chain_ids < 0):
        raise ValueError("cyclic temporal chains are unsupported")
    return TemporalChains(chain_ids, positions, tuple(chains))


class ChainAwareSolarWindDataset(Dataset):
    def __init__(self, *args, temporal_chains, **kwargs):
        from dataset import SolarWindDataset

        self.base_dataset = SolarWindDataset(*args, **kwargs)
        self.indexes = self.base_dataset.indexes
        if len(temporal_chains.chain_ids) != len(self.base_dataset.image_indexes):
            raise ValueError("chain metadata and input rows have different lengths")
        self.chain_ids = temporal_chains.chain_ids
        self.chain_positions = temporal_chains.positions

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, item):
        result = self.base_dataset[item]
        row_index = int(self.indexes[item])
        result["row_index"] = row_index
        result["chain_id"] = int(self.chain_ids[row_index])
        result["chain_position"] = int(self.chain_positions[row_index])
        return result


def make_chain_loader(dataset, training, chain_balanced=True):
    sampler = None
    shuffle = False
    if training and chain_balanced:
        local_chain_ids = dataset.chain_ids[dataset.indexes]
        unique_ids, counts = np.unique(local_chain_ids, return_counts=True)
        count_by_id = dict(zip(unique_ids.tolist(), counts.tolist()))
        weights = np.asarray(
            [1.0 / count_by_id[int(chain_id)] for chain_id in local_chain_ids],
            dtype=np.float64,
        )
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights),
            num_samples=len(dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(SEED),
        )
    elif training:
        shuffle = True

    options = dict(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        generator=torch.Generator().manual_seed(SEED),
    )
    if NUM_WORKERS > 0:
        options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**options)


def chain_manifest(temporal_chains):
    lengths = temporal_chains.lengths
    return {
        "chain_count": int(temporal_chains.count),
        "row_count": int(lengths.sum()),
        "min_length": int(lengths.min()),
        "max_length": int(lengths.max()),
        "mean_length": float(lengths.mean()),
        "lengths": lengths.tolist(),
    }
