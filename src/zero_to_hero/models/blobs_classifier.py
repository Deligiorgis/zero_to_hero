"""
Model that classifies blobs
"""
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import cm
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import nn

from zero_to_hero.logging import get_data_from_outputs
from zero_to_hero.metrics import compute_metrics
from zero_to_hero.models.models import MLP


class BlobsClassifierModel(pl.LightningModule):  # pylint: disable=too-many-ancestors
    # the ancestors come from PyTorch Lightning
    """
    Implementation of the Blobs' Classifier
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        self.config = config

        self.criterion = nn.CrossEntropyLoss(reduction="none")

        self.mlp = MLP(
            in_features=self.config["data"]["in_features"],
            list_out_features=self.config["model"]["out_features"],
            list_linear_dropout=self.config["model"]["dropout"],
        )
        self.softmax = nn.Softmax(dim=1)

        self.example_input_array = torch.rand(10, self.config["data"]["in_features"])

        self.save_hyperparameters()

    def forward(self, data: torch.Tensor) -> torch.Tensor:  # type: ignore # pylint: disable=arguments-differ
        # Signature of "forward" incompatible with supertype "LightningModule"
        # No need for all arguments
        return self.mlp(data.float())

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.config["hyper_parameters"]["learning_rate"],
        )

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
            predictions = outputs.argmax(dim=1)
        return {
            "loss": losses.mean(),
            "losses": losses.detach(),
            "data": data.detach(),
            "predictions": predictions.detach(),
            "targets": targets.detach(),
        }

    def get_figure(self, data: torch.Tensor, targets: torch.Tensor) -> plt.Figure:
        """
        Get the figure that is needed in TensorBoard
        :param data: torch.Tensor, coordinates of the blobs
        :param targets: torch.Tensor, target labels of each point
        :return: matplotlib figure needed from TensorBoard
        """
        fig = plt.figure()
        colors = cm.get_cmap("rainbow")(np.linspace(0, 1, torch.unique(targets).shape[0]))
        for enum, color in enumerate(colors):
            plt.scatter(
                data[targets.squeeze() == enum][:, 0].cpu().numpy(),
                data[targets.squeeze() == enum][:, 1].cpu().numpy(),
                color=color,
            )
        boundary = torch.linspace(-2.5, 2.5, 50)
        xx_mesh, yy_mesh = torch.meshgrid(boundary, boundary)
        self.train(mode=False)
        with torch.no_grad():
            zz_mesh = (
                self(torch.vstack([xx_mesh.reshape(1, -1), yy_mesh.reshape(1, -1)]).T.to(self.device))
                .argmax(dim=1)
                .cpu()
            )
        plt.imshow(
            X=np.rot90(zz_mesh.reshape(xx_mesh.shape).numpy(), k=1),
            extent=(xx_mesh.min(), xx_mesh.max(), yy_mesh.min(), yy_mesh.max()),
            cmap="rainbow",
            alpha=0.5,
        )
        plt.grid()
        plt.tight_layout()
        return fig

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        training_loss, training_accuracy = compute_metrics(outputs=outputs, device=self.device)
        self.log("loss", {"train": training_loss}, on_epoch=True)
        self.log("accuracy", {"train": training_accuracy}, on_epoch=True)

        if self.trainer is not None:
            self.log("step", self.trainer.current_epoch)
            assert isinstance(outputs[0], dict)
            if self.trainer.current_epoch % 25 == 0:
                dict_data = get_data_from_outputs(
                    keys=["data", "targets", "predictions"],
                    outputs=outputs,
                )

                tensorboard = self.logger.experiment
                tensorboard.add_pr_curve(
                    tag="train-blobs-pr-curve",
                    labels=dict_data["targets"].cpu().squeeze(),
                    predictions=dict_data["predictions"].cpu().squeeze(),
                    global_step=self.trainer.current_epoch,
                )

                if outputs[0]["data"].shape[1] == 2:
                    fig = self.get_figure(dict_data["data"], dict_data["targets"])
                    tensorboard.add_figure(tag="train", figure=fig, global_step=self.trainer.current_epoch)

    def validation_step(  # type: ignore # pylint: disable=arguments-differ
        # Signature of "validation_step" incompatible with supertype "LightningModule"
        # No need for all arguments
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        _: int,
    ) -> Optional[STEP_OUTPUT]:
        return self.training_step(batch=batch, _=_)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        validation_loss, validation_accuracy = compute_metrics(outputs=outputs, device=self.device)
        if self.trainer is not None:
            self.log("step", self.trainer.current_epoch)
        self.log("loss", {"valid": validation_loss}, on_epoch=True)
        self.log("accuracy", {"valid": validation_accuracy}, on_epoch=True)
        self.log("validation_loss", validation_loss, on_epoch=True)

        if self.trainer is not None:
            self.log("step", self.trainer.current_epoch)
            assert isinstance(outputs[0], dict)
            if self.trainer.current_epoch % 25 == 0:
                dict_data = get_data_from_outputs(
                    keys=["data", "targets", "predictions"],
                    outputs=outputs,
                )

                tensorboard = self.logger.experiment
                tensorboard.add_pr_curve(
                    tag="valid-blobs-pr-curve",
                    labels=dict_data["targets"].cpu().squeeze(),
                    predictions=dict_data["predictions"].cpu().squeeze(),
                    global_step=self.trainer.current_epoch,
                )

                if outputs[0]["data"].shape[1] == 2:
                    fig = self.get_figure(dict_data["data"], dict_data["targets"])
                    tensorboard.add_figure(tag="valid", figure=fig, global_step=self.trainer.current_epoch)

    def test_step(  # type: ignore # pylint: disable=arguments-differ
        # Signature of "test_step" incompatible with supertype "LightningModule"
        # No need for all arguments
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        _: int,
    ) -> Optional[STEP_OUTPUT]:
        return self.training_step(batch=batch, _=_)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        test_loss, test_accuracy = compute_metrics(outputs=outputs, device=self.device)
        self.log("loss", test_loss, on_epoch=True)
        self.log("accuracy", test_accuracy, on_epoch=True)

        if self.trainer is not None:
            dict_data = get_data_from_outputs(
                keys=["data", "targets", "predictions"],
                outputs=outputs,
            )

            tensorboard = self.logger.experiment
            tensorboard.add_pr_curve(
                tag="test-blobs-pr-curve",
                labels=dict_data["targets"].cpu().squeeze(),
                predictions=dict_data["predictions"].cpu().squeeze(),
                global_step=self.trainer.current_epoch,
            )

            tensorboard.add_embedding(
                tag="test-blobs-embedding-space",
                mat=dict_data["data"].cpu(),
                metadata=dict_data["targets"].cpu(),
                global_step=self.trainer.current_epoch,
            )

            if outputs[0]["data"].shape[1] == 2:
                fig = self.get_figure(dict_data["data"], dict_data["targets"])
                tensorboard.add_figure(tag="test", figure=fig, global_step=self.trainer.current_epoch)

    def predict_step(  # type: ignore  # Signature of "predict_step" incompatible with supertype "LightningModule"
        self,
        batch: torch.Tensor,
        _: int,
        __: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        probabilities = self.softmax(self(batch))
        return {
            "predictions": probabilities.argmax(dim=1).cpu(),
            "probabilities": probabilities.cpu(),
        }
