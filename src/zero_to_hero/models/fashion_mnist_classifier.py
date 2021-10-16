"""
FashionMNIST classifier
"""
import operator
from functools import reduce
from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT

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

        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")

        self.embeds = torch.empty(1)

        self.cnn_layers = CNN(
            in_channels=self.configs["data"]["in_channels"],
            list_out_channels=self.configs["model"]["convolutional"]["out_channels"],
            list_kernel_size=self.configs["model"]["convolutional"]["kernel_size"],
            list_cnn_dropout=self.configs["model"]["convolutional"]["cnn_dropout"],
            dim=2,
            activation_as_last_layer=False,
        )

        self.pooling_layer = torch.nn.MaxPool2d(kernel_size=self.configs["model"]["pooling"]["kernel_size"])

        self.example_input_array = torch.rand(10, 1, 28, 28)
        in_features = reduce(operator.mul, self.pooling_layer(self.cnn_layers(self.example_input_array)).shape[1:], 1)

        self.linear_layers = MLP(
            in_features=in_features,
            list_out_features=self.configs["model"]["linear"]["out_features"],
            list_linear_dropout=self.configs["model"]["linear"]["dropout"],
        )

        self.save_hyperparameters()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.configs["hyper_parameters"]["learning_rate"],
            weight_decay=self.configs["hyper_parameters"]["weight_decay"],
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:  # type: ignore # pylint: disable=arguments-differ
        # Signature of "forward" incompatible with supertype "LightningModule"
        # No need for all arguments
        self.embeds = self.cnn_layers(data)
        return self.linear_layers(self.pooling_layer(self.embeds).view(self.embeds.shape[0], -1))

    def training_step(  # type: ignore # pylint: disable=arguments-differ
        # Signature of "forward" incompatible with supertype "LightningModule"
        # No need for all arguments
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        _: int,
    ) -> STEP_OUTPUT:
        data, targets = batch
        outputs = self(data)
        losses = self.criterion(outputs, targets)
        with torch.no_grad():
            predictions = outputs.detach().argmax(dim=1)
        return {
            "loss": losses.mean(),
            "losses": losses.detach().cpu(),
            "data": data.detach().cpu(),
            "predictions": predictions.detach().cpu(),
            "targets": targets.detach().cpu(),
            "embeds": self.embeds.detach().cpu(),
        }

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        training_loss, training_accuracy = compute_metrics(outputs=outputs, device=self.device)
        self.log("loss", {"train": training_loss}, on_epoch=True)
        self.log("accuracy", {"train": training_accuracy}, on_epoch=True)

        if self.trainer is not None:
            self.log("step", self.trainer.current_epoch)

    def validation_step(  # type: ignore # pylint: disable=arguments-differ
        # Signature of "training_step" incompatible with supertype "LightningModule"
        # No need for all arguments
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        _: int,
    ) -> Optional[STEP_OUTPUT]:
        return self.training_step(batch=batch, _=_)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        validation_loss, validation_accuracy = compute_metrics(outputs=outputs, device=self.device)
        self.log("loss", {"valid": validation_loss}, on_epoch=True)
        self.log("accuracy", {"valid": validation_accuracy}, on_epoch=True)
        self.log("validation_loss", validation_loss, on_epoch=True)

        if self.trainer is not None:
            self.log("step", self.trainer.current_epoch)
            assert isinstance(outputs[0], dict)

            if self.trainer.current_epoch % 20 == 0 and self.trainer.current_epoch > 0:
                dict_data = get_data_from_outputs(
                    keys=["data", "targets", "predictions", "embeds"],
                    outputs=outputs,
                )

                val_indices = self.trainer.datamodule.val_dataloader().dataset.indices
                val_dataset = self.trainer.datamodule.val_dataloader().dataset.dataset
                class_to_idx = {v: k for k, v in val_dataset.class_to_idx.items()}
                metadata = list(map(lambda target: class_to_idx[target.item()], dict_data["targets"].cpu()))

                tensorboard = self.logger.experiment
                tensorboard.add_embedding(
                    tag="valid-fashion-mnist-embedding-space",
                    mat=dict_data["embeds"].cpu().view(dict_data["embeds"].shape[0], -1),
                    metadata=metadata,
                    global_step=self.trainer.current_epoch,
                    label_img=val_dataset.data.unsqueeze(1)[val_indices],
                )

    def test_step(  # type: ignore # pylint: disable=arguments-differ
        # Signature of "training_step" incompatible with supertype "LightningModule"
        # No need for all arguments
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        _: int,
    ) -> Optional[STEP_OUTPUT]:
        return self.training_step(batch=batch, _=_)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        test_loss, test_accuracy = compute_metrics(outputs=outputs, device=self.device)
        self.log("test_loss", test_loss)
        self.log("test_accuracy", test_accuracy)

        dict_data = get_data_from_outputs(
            keys=["data", "targets", "predictions", "embeds"],
            outputs=outputs,
        )

        tensorboard = self.logger.experiment
        if self.trainer is not None:
            test_dataset = self.trainer.datamodule.test_dataloader().dataset
            class_to_idx = {v: k for k, v in test_dataset.class_to_idx.items()}
            metadata = list(map(lambda target: class_to_idx[target.item()], dict_data["targets"].cpu()))
            tensorboard.add_embedding(
                tag="test-fashion-mnist-embedding-space",
                mat=dict_data["embeds"].cpu().view(dict_data["embeds"].shape[0], -1),
                metadata=metadata,
                label_img=test_dataset.data.unsqueeze(1),
            )
