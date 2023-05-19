from typing import Optional, List

import os
import numpy as np
import torch
from torch.utils.data import Dataset


def random_shift(data, target_len: int):
    if len(data) == target_len:
        return data, 0
    rand_start_idx = np.random.randint(len(data) - target_len)
    idxs = rand_start_idx + np.arange(0, target_len)
    return np.take_along_axis(data, idxs, axis=-1), rand_start_idx


def random_phase(data):
    rand_phase = np.random.rand()
    return data * np.exp(1j * 2 * np.pi * (rand_phase + 0j)), rand_phase


class RFDatasetBase(Dataset):
    def __init__(
        self,
        root_dir: str,
        target_len: Optional[int] = None,
        augmentation: bool = False,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None
    ):
        super().__init__()
        self.root_dir = root_dir
        self.data = [s for s in os.listdir(root_dir) if s.endswith(".npy")]
        self.target_len = target_len
        self.augmentation = augmentation

        self.normalize = mean and std
        if self.normalize:
            self.mean = torch.tensor(mean).reshape(1, 2)
            self.std = torch.tensor(std).reshape(1, 2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = np.load(os.path.join(self.root_dir, self.data[idx]))
        if self.augmentation:
            sample, shift = random_shift(sample, self.target_len)
            sample, phase = random_phase(sample)
            if self.normalize:
                sample = torch.view_as_real(torch.tensor(sample))
                sample = torch.view_as_complex((sample - self.mean) / self.std).numpy()
            return {
                "sample": torch.view_as_real(torch.tensor(sample)).transpose(0, 1).float(),
                "shift": torch.tensor(shift).float(),
                "phase": torch.tensor(phase).float(),
            }
        else:
            if self.normalize:
                sample = torch.view_as_real(sample)
                sample = torch.view_as_complex((sample - self.mean) / self.std)
            return {
                "sample": torch.view_as_real(torch.tensor(sample)).transpose(0, 1).float(),
            }


def get_train_val_dataset(dataset: Dataset, train_fraction: float):
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_fraction, 1 - train_fraction], generator=torch.Generator().manual_seed(42))
    return train_dataset, val_dataset
