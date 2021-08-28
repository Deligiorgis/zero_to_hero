"""
Model that classifies blobs
"""
from typing import List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import nn


class BlobsClassifierModel(pl.LightningModule):  # pylint: disable=too-many-ancestors
    # the ancestors come from PyTorch Lightning
    """
    Implementation of the Blobs' Classifier
    """

    def __init__(
        self,
        n_features: int,
        n_centers: int,
        hidden_nodes: Optional[List[int]] = None,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()

        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.learning_rate = learning_rate

        if hidden_nodes is not None:
            layers = [
                nn.Linear(in_features=n_features, out_features=hidden_nodes[0]),
                nn.ReLU(),
            ]
            for hidden in hidden_nodes[1:-1]:
                layers.extend([nn.Linear(in_features=hidden, out_features=hidden), nn.ReLU()])
            layers.append(nn.Linear(in_features=hidden_nodes[-1], out_features=n_centers))
        else:
            layers = [
                nn.Linear(in_features=n_features, out_features=n_centers),
                nn.ReLU(),
            ]

        self.layers = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data: torch.Tensor) -> torch.Tensor:  # type: ignore # pylint: disable=arguments-differ
        # Signature of "forward" incompatible with supertype "LightningModule"
        # No need for all arguments
        return self.layers(data.float())

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            lr=self.learning_rate,
            params=self.parameters(),
        )

    @staticmethod
    def compute_metrics(
        outputs: EPOCH_OUTPUT,
        device: Union[str, torch.device] = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute total metrics
        :param outputs: EPOCH_OUTPUT
        :param device: device on which the computations are executed
        :return: average loss: float, average accuracy: float
        """
        total_loss, accuracy = torch.zeros(1, device=device), torch.zeros(1, device=device)
        n_samples = 0
        for output in outputs:
            assert isinstance(output, dict)
            total_loss += output["losses"].sum()
            accuracy += (output["predictions"] == output["targets"].squeeze()).sum()
            n_samples += output["losses"].shape[0]
        return total_loss / n_samples, accuracy / n_samples

    def training_step(  # type: ignore # pylint: disable=arguments-differ
        # Signature of "training_step" incompatible with supertype "LightningModule"
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        _: int,
    ) -> STEP_OUTPUT:
        data, targets = batch
        outputs = self(data)
        losses = self.criterion(outputs, targets.squeeze().long())
        with torch.no_grad():
            predictions = self.softmax(outputs).argmax(dim=1)
        return {
            "loss": losses.mean(),
            "losses": losses.detach(),
            "predictions": predictions.detach(),
            "targets": targets.detach(),
        }

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        training_loss, training_accuracy = self.compute_metrics(outputs=outputs, device=self.device)
        if self.trainer is not None:
            self.log("step", self.trainer.current_epoch)
        self.log("loss", {"train": training_loss}, on_epoch=True)
        self.log("accuracy", {"train": training_accuracy}, on_epoch=True)

    def validation_step(  # type: ignore # pylint: disable=arguments-differ
        # Signature of "validation_step" incompatible with supertype "LightningModule"
        # No need for all arguments
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        _: int,
    ) -> Optional[STEP_OUTPUT]:
        return self.training_step(batch=batch, _=_)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        validation_loss, validation_accuracy = self.compute_metrics(outputs=outputs, device=self.device)
        if self.trainer is not None:
            self.log("step", self.trainer.current_epoch)
        self.log("loss", {"val": validation_loss}, on_epoch=True)
        self.log("accuracy", {"val": validation_accuracy}, on_epoch=True)
        self.log("validation_loss", validation_loss)

    def test_step(  # type: ignore # pylint: disable=arguments-differ
        # Signature of "test_step" incompatible with supertype "LightningModule"
        # No need for all arguments
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        _: int,
    ) -> Optional[STEP_OUTPUT]:
        return self.training_step(batch=batch, _=_)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        test_loss, test_accuracy = self.compute_metrics(outputs=outputs, device=self.device)
        self.log("loss", test_loss, on_epoch=True)
        self.log("accuracy", test_accuracy, on_epoch=True)

    def predict_step(  # type: ignore  # Signature of "predict_step" incompatible with supertype "LightningModule"
        self,
        batch: torch.Tensor,
        _: int,
        __: Optional[int] = None,
    ) -> torch.Tensor:
        return self.softmax(self(batch))
