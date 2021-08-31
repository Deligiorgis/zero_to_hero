"""
DataModule for Blobs
"""
from typing import Any, Dict, Tuple

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from sklearn.datasets import make_blobs
from torch.utils.data import DataLoader, Dataset

from zero_to_hero.data.blobs.dataset import BlobsDataset
from zero_to_hero.data.statistics import standardize_step


class BlobsDataModule(pl.LightningDataModule):
    """
    Implementation of the Blobs' PyTorch Lightning DataModule
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        self.config = config

        train_n_samples = int(self.config["data"]["n_samples"] * self.config["data"]["train_ratio"])
        valid_n_samples = int(self.config["data"]["n_samples"] * self.config["data"]["valid_ratio"])
        test_n_samples = self.config["data"]["n_samples"] - train_n_samples - valid_n_samples

        n_centers = len(self.config["data"]["centers"])

        train_data, train_targets = self.get_data_and_targets(int(train_n_samples / n_centers))
        valid_data, valid_targets = self.get_data_and_targets(int(valid_n_samples / n_centers))
        test_data, test_targets = self.get_data_and_targets(int(test_n_samples / n_centers))

        train_data, valid_data, test_data = standardize_step(  # type: ignore # False positive mypy
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
        )

        self.train_dataset = BlobsDataset(
            data=train_data,
            targets=train_targets,
        )
        self.valid_dataset = BlobsDataset(
            data=valid_data,
            targets=valid_targets,
        )
        self.test_dataset = BlobsDataset(
            data=test_data,
            targets=test_targets,
        )

    def get_data_and_targets(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the feature data and targets (blobs) for each given center.
        :param n_samples: int, number of blobs to sample
        :return: (features, target)
        """
        data, targets = [], []
        for enum, center in enumerate(self.config["data"]["centers"]):
            print(
                f"Label: {enum} "
                f"Center: {center} "
                f"Number of samples: {n_samples} "
                f"Number of features: {self.config['data']['n_features']}"
            )
            features, target, *_ = make_blobs(
                n_samples=n_samples,
                n_features=self.config["data"]["n_features"],
                centers=[center],
                cluster_std=self.config["data"]["cluster_std"],
                shuffle=False,
                random_state=1,
            )
            target += enum
            data.extend(features.tolist())
            targets.extend(target.tolist())
        return np.vstack(data), np.vstack(targets)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        assert isinstance(self.train_dataset, Dataset)
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config["hyper_parameters"]["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=self.config["hyper_parameters"]["num_workers"],
            drop_last=False,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        assert isinstance(self.valid_dataset, Dataset)
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.config["hyper_parameters"]["batch_size"],
            shuffle=False,
            pin_memory=True,
            num_workers=self.config["hyper_parameters"]["num_workers"],
            drop_last=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        assert isinstance(self.test_dataset, Dataset)
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config["hyper_parameters"]["batch_size"],
            shuffle=False,
            pin_memory=True,
            num_workers=self.config["hyper_parameters"]["num_workers"],
            drop_last=False,
        )
