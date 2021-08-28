"""
Dataset for Blobs
"""
from typing import Tuple

import numpy as np
from torch.utils.data import Dataset


class BlobsDataset(Dataset):
    """
    Implementation of the Blobs' PyTorch Dataset
    """

    def __init__(self, data: np.ndarray, targets: np.ndarray) -> None:
        """
        Initialize Blobs' Dataset
        :param data: np.ndarray with shape (n_samples, n_features)
        :param targets: np.ndarray with shape (n_samples, 1)
        """
        self.size = data.shape[0]
        assert self.size == targets.shape[0], "Number of input data is not equal to the number of targets"
        self.data = data
        self.targets = targets

    def __len__(self) -> int:
        """

        :return: n_samples
        """
        return self.size

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """

        :param idx: int received from sampler
        :return: (datum, target)
        """
        return self.data[idx], self.targets[idx]
