"""
FashionMNIST classifier
"""
import operator
from functools import reduce
from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import nn

from zero_to_hero.logging import get_data_from_outputs
from zero_to_hero.metrics import compute_metrics
from zero_to_hero.models.models import CNN, MLP


class FashionMNISTClassifier(pl.LightningModule):  # pylint: disable=too-many-ancestors
    # the ancestors come from PyTorch Lightning
    """
    Implementation of the model that classifies the FashionMNIST clothes
    """

    def __init__(self, configs: Dict) -> None:
        super().__init__()

        self.configs = configs

        self.criterion = nn.CrossEntropyLoss(reduction="none")

        self.embeds = torch.empty(1)

        self.cnn_layers = CNN(
            in_channels=self.configs["model"]["in_channels"],
            list_out_channels=self.configs["model"]["out_channels"],
            list_kernel_size=self.configs["model"]["kernel_size"],
        )

        self.example_input_array = torch.rand(10, 1, 28, 28)
        n_features = reduce(operator.mul, self.cnn_layers(self.example_input_array).shape[1:], 1)

        self.mlp = MLP(
            n_targets=self.configs["data"]["n_targets"],
            n_features=n_features,
            hidden_nodes=self.configs["model"]["hidden_nodes"],
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.configs["hyper_parameters"]["learning_rate"],
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:  # type: ignore # pylint: disable=arguments-differ
        # Signature of "forward" incompatible with supertype "LightningModule"
        # No need for all arguments
        self.embeds = self.cnn_layers(data)
        return self.mlp(self.embeds.view(self.embeds.shape[0], -1))

    def training_step(  # type: ignore # pylint: disable=arguments-differ
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        _: int,
    ) -> STEP_OUTPUT:
        # Signature of "forward" incompatible with supertype "LightningModule"
        # No need for all arguments
        data, targets = batch
        outputs = self(data)
        losses = self.criterion(outputs, targets)
        with torch.no_grad():
            predictions = outputs.detach().argmax(dim=1)
        return {
            "loss": losses.mean(),
            "losses": losses.detach(),
            "data": data.detach(),
            "predictions": predictions.detach(),
            "targets": targets.detach(),
            "embeds": self.embeds.detach(),
        }

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        training_loss, training_accuracy = compute_metrics(outputs=outputs, device=self.device)
        self.log("loss", {"train": training_loss}, on_epoch=True)
        self.log("accuracy", {"train": training_accuracy}, on_epoch=True)

        if self.trainer is not None:
            self.log("step", self.trainer.current_epoch)
            assert isinstance(outputs[0], dict)
            if self.trainer.current_epoch % 20 == 0:
                dict_data = get_data_from_outputs(
                    keys=["data", "targets", "predictions", "embeds"],
                    outputs=outputs,
                )

                self.logger.experiment.add_pr_curve(
                    tag="train-fashion-mnist-pr-curve",
                    labels=dict_data["targets"].cpu().squeeze(),
                    predictions=dict_data["predictions"].cpu().squeeze(),
                    global_step=self.trainer.current_epoch,
                )

                self.logger.experiment.add_embedding(
                    tag="train-fashion-mnist-embedding-space",
                    mat=dict_data["embeds"].cpu().view(dict_data["embeds"].shape[0], -1),
                    metadata=dict_data["targets"].cpu(),
                    global_step=self.trainer.current_epoch,
                )

    def validation_step(  # type: ignore # pylint: disable=arguments-differ
        # Signature of "training_step" incompatible with supertype "LightningModule"
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        _: int,
    ) -> Optional[STEP_OUTPUT]:
        return self.training_step(batch=batch, _=_)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        validation_loss, validation_accuracy = compute_metrics(outputs=outputs, device=self.device)
        if self.trainer is not None:
            self.log("step", self.trainer.current_epoch)
        self.log("loss", {"val": validation_loss}, on_epoch=True)
        self.log("accuracy", {"val": validation_accuracy}, on_epoch=True)
        self.log("validation_loss", validation_loss)

        if self.trainer is not None:
            self.log("step", self.trainer.current_epoch)
            assert isinstance(outputs[0], dict)
            if self.trainer.current_epoch % 20 == 0:
                dict_data = get_data_from_outputs(
                    keys=["data", "targets", "predictions", "embeds"],
                    outputs=outputs,
                )

                self.logger.experiment.add_pr_curve(
                    tag="valid-fashion-mnist-pr-curve",
                    labels=dict_data["targets"].cpu().squeeze(),
                    predictions=dict_data["predictions"].cpu().squeeze(),
                    global_step=self.trainer.current_epoch,
                )

                self.logger.experiment.add_embedding(
                    tag="valid-fashion-mnist-embedding-space",
                    mat=dict_data["embeds"].cpu().view(dict_data["embeds"].shape[0], -1),
                    metadata=dict_data["targets"].cpu(),
                    global_step=self.trainer.current_epoch,
                )

    def test_step(  # type: ignore # pylint: disable=arguments-differ
        # Signature of "training_step" incompatible with supertype "LightningModule"
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        _: int,
    ) -> Optional[STEP_OUTPUT]:
        return self.training_step(batch=batch, _=_)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        test_loss, test_accuracy = compute_metrics(outputs=outputs, device=self.device)
        self.log("loss", {"test": test_loss}, on_epoch=True)
        self.log("accuracy", {"test": test_accuracy}, on_epoch=True)

        if self.trainer is not None:
            self.log("step", self.trainer.current_epoch)

            if self.trainer.current_epoch % 20 == 0:
                dict_data = get_data_from_outputs(
                    keys=["data", "targets", "predictions"],
                    outputs=outputs,
                )

                self.logger.experiment.add_pr_curve(
                    tag="test-fashion-mnist-pr-curve",
                    labels=dict_data["targets"].cpu().squeeze(),
                    predictions=dict_data["predictions"].cpu().squeeze(),
                    global_step=self.trainer.current_epoch,
                )

                self.logger.experiment.add_embedding(
                    tag="test-fashion-mnist-embedding-space",
                    mat=dict_data["embeds"].cpu().view(dict_data["embeds"].shape[0], -1),
                    metadata=dict_data["targets"].cpu(),
                    global_step=self.trainer.current_epoch,
                )
