"""
Dataset & DataModule for Blobs
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from sklearn.datasets import make_blobs
from torch.utils.data import DataLoader, Dataset

from zero_to_hero.data.statistics import standardize_step


class UnpackingError(Exception):
    """
    Customized Exception Class for failures during unpackaging of objects
    """

    def __init__(self, fault_object: Any, message: str = "Unpacking of object failed.") -> None:
        self.message = f"{message} Length of object: {len(fault_object)}."
        super().__init__(message)


class BlobsDataset(Dataset):
    """
    Implementation of the Blobs' PyTorch Dataset
    """

    def __init__(self, data: np.ndarray, targets: Optional[np.ndarray] = None) -> None:
        """
        Initialize Blobs' Dataset
        :param data: np.ndarray with shape (n_samples, n_features)
        :param targets: np.ndarray with shape (n_samples, 1)
        """
        self.data = data
        self.size = self.data.shape[0]
        if targets is not None:
            assert self.size == targets.shape[0], "Number of input data is not equal to the number of targets"
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
        if self.targets is not None:
            return self.data[idx], self.targets[idx]
        return self.data[idx]


class BlobsDataModule(pl.LightningDataModule):
    """
    Implementation of the Blobs' PyTorch Lightning DataModule
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        self.config = config

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.predict_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        assert stage in (None, "fit", "validate", "test", "predict")
        train_n_samples = int(self.config["data"]["n_samples"] * self.config["data"]["train_ratio"])
        valid_n_samples = int(self.config["data"]["n_samples"] * self.config["data"]["valid_ratio"])
        test_n_samples = self.config["data"]["n_samples"] - train_n_samples - valid_n_samples

        n_centers = len(self.config["data"]["centers"])

        print("Training data & targets.")
        train_data, train_targets = self.get_data_and_targets(int(train_n_samples / n_centers))
        print("Validation data & targets.")
        valid_data, valid_targets = self.get_data_and_targets(int(valid_n_samples / n_centers))
        print("Testing data & targets.")
        test_data, test_targets = self.get_data_and_targets(int(test_n_samples / n_centers))
        print("Predicting data & targets.")
        predict_data, _ = self.get_data_and_targets(n_centers * 4)

        assert isinstance(predict_data, np.ndarray)
        standardized_data = standardize_step(  # type: ignore # False positive mypy
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
            predict_data=predict_data,
        )
        if len(standardized_data) == 3:
            (  # pylint: disable=unbalanced-tuple-unpacking
                # Possible error is going to be handled by the Custom Exception UnpackingError
                train_data,
                valid_data,
                test_data,
            ) = standardized_data  # type: ignore # False positive.
        elif len(standardized_data) == 4:
            (  # pylint: disable=unbalanced-tuple-unpacking
                # Possible error is going to be handled by the Custom Exception UnpackingError
                train_data,
                valid_data,
                test_data,
                predict_data,
            ) = standardized_data  # type: ignore # False positive.
        else:
            raise UnpackingError(standardized_data)

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
        self.predict_dataset = BlobsDataset(
            data=predict_data,
            targets=None,
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
                f"Number of features: {self.config['data']['in_features']}"
            )
            features, target, *_ = make_blobs(
                n_samples=n_samples,
                n_features=self.config["data"]["in_features"],
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
            pin_memory=self.config["hyper_parameters"]["pin_memory"],
            num_workers=self.config["hyper_parameters"]["num_workers"],
            drop_last=False,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        assert isinstance(self.valid_dataset, Dataset)
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.config["hyper_parameters"]["batch_size"],
            shuffle=False,
            pin_memory=self.config["hyper_parameters"]["pin_memory"],
            num_workers=self.config["hyper_parameters"]["num_workers"],
            drop_last=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        assert isinstance(self.test_dataset, Dataset)
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config["hyper_parameters"]["batch_size"],
            shuffle=False,
            pin_memory=self.config["hyper_parameters"]["pin_memory"],
            num_workers=self.config["hyper_parameters"]["num_workers"],
            drop_last=False,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        assert isinstance(self.predict_dataset, Dataset)
        return DataLoader(
            dataset=self.predict_dataset,
            batch_size=self.config["hyper_parameters"]["batch_size"],
            shuffle=False,
            pin_memory=self.config["hyper_parameters"]["pin_memory"],
            num_workers=self.config["hyper_parameters"]["num_workers"],
            drop_last=False,
        )
