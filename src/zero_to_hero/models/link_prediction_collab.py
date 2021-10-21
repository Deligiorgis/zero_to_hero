"""
SEAL implementation for link-prediction
"""
from typing import Dict, Optional, Tuple

import dgl
import pytorch_lightning as pl
import torch
from ogb.linkproppred import Evaluator
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch.nn import functional as F

from zero_to_hero.models.models import DGCNN


class LinkPredictorCollab(pl.LightningModule):  # pylint: disable=too-many-ancestors
    # the ancestors come from PyTorch Lightning
    """
    SEAL (learning from Subgraphs, Embeddings and Attributes for Link prediction) model implementation
    by inheriting the PyTorch-Lightning Module
    """

    def __init__(self, ndata: torch.Tensor, edata: torch.Tensor, config: Dict) -> None:
        super().__init__()

        self.config = config

        self.model = DGCNN(
            ndata=ndata,
            edata=edata,
            config=self.config,
        )

        self.sigmoid = torch.nn.Sigmoid()

        self.pos_weight = torch.tensor([3.0])

        self.save_hyperparameters(ignore=["ndata", "edata"])

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config["hyper_parameters"]["learning_rate"],
            weight_decay=2.3e-6,
        )

    def forward(  # type: ignore # pylint: disable=arguments-differ
        # Signature of "forward" incompatible with supertype "LightningModule"
        # No need for all arguments
        self,
        graph: dgl.DGLHeteroGraph,
    ) -> torch.Tensor:
        return self.model(graph, graph.ndata["label"], graph.ndata[dgl.NID], graph.edata[dgl.EID])

    def training_step(  # type: ignore # pylint: disable=arguments-differ
        # Signature of "training_step" incompatible with supertype "LightningModule"
        # No need for all arguments
        self,
        batch: Tuple[dgl.DGLHeteroGraph, torch.Tensor],
        _: int,
    ) -> STEP_OUTPUT:
        graph, target_links = batch
        predicted_links = self(graph)
        loss = F.binary_cross_entropy_with_logits(
            input=predicted_links,
            target=target_links,
            reduction="mean",
            pos_weight=self.pos_weight.to(self.device),
        )
        return {
            "loss": loss,
            "links": predicted_links.detach().cpu(),
            "targets": target_links.detach().cpu(),
        }

    @staticmethod
    @torch.no_grad()
    def compute_accuracy(pos_predictions: torch.Tensor, neg_predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Method that implements the following accuracy metrics:
         * Positive accuracy
         * Negative accuracy
         * Total accuracy

        This method runs with torch.no_grad()

        :param pos_predictions: Positive predicted links after applying the sigmoid function
        :param neg_predictions: Negative predicted links after applying the sigmoid function
        :return: Dictionary with the accuracy metrics
        """
        pos_acc = (pos_predictions >= 0.5).float().mean()
        neg_acc = (neg_predictions < 0.5).float().mean()
        total_acc = ((pos_predictions >= 0.5).float().sum() + (neg_predictions < 0.5).float().sum()) / (
            pos_predictions.shape[0] + neg_predictions.shape[0]
        )
        return {
            "pos_acc": pos_acc,
            "neg_acc": neg_acc,
            "total_acc": total_acc,
        }

    @staticmethod
    @torch.no_grad()
    def compute_hits(pos_predictions: torch.Tensor, neg_predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computing Hits@50 the metric which has been decided to use in OGBL-Collab's Leaderboard

        :param pos_predictions: Positive predictions
        :param neg_predictions: Negative predictions
        :return: Hits@50
        """
        evaluator = Evaluator(name="ogbl-collab")
        return evaluator.eval(
            {
                "y_pred_pos": pos_predictions.squeeze(),
                "y_pred_neg": neg_predictions.squeeze(),
            }
        )

    @torch.no_grad()
    def compute_loss(
        self, links: torch.Tensor, targets: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Method that estimates the following:
         * Positive loss
         * Negative loss
         * Total loss

        This method runs with torch.no_grad()

        :param links: predicted links
        :param targets: target links
        :param pos_mask: which links should be positive based on targets
        :param neg_mask: which links should be negative based on targets
        :return:
        """
        pos_loss = F.binary_cross_entropy_with_logits(
            input=links[pos_mask],
            target=targets[pos_mask],
            pos_weight=self.pos_weight.to(links.device),
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            input=links[neg_mask],
            target=targets[neg_mask],
            pos_weight=self.pos_weight.to(links.device),
        )
        total_loss = F.binary_cross_entropy_with_logits(
            input=links,
            target=targets,
            pos_weight=self.pos_weight.to(links.device),
        )
        return {
            "pos_loss": pos_loss,
            "neg_loss": neg_loss,
            "total_loss": total_loss,
        }

    @torch.no_grad()
    def evaluate_metrics(self, outputs: EPOCH_OUTPUT) -> Dict[str, torch.Tensor]:
        """
        Evaluation method which computes the following metrics:
         * Positive loss, Negative loss, Total loss
         * Positive accuracy, Negative accuracy, Total accuracy
         * Hits@50

        The method runs with torch.no_grad()

        :param outputs: EPOCH_OUTPUT
        :return: Dict[str, torch.Tensor]
        """
        assert isinstance(outputs[0], dict)
        links = outputs[0]["links"]
        targets = outputs[0]["targets"]
        for output in outputs[1:]:
            assert isinstance(output, dict)
            links = torch.cat([links, output["links"]])
            targets = torch.cat([targets, output["targets"]])

        pos_mask = targets == 1
        neg_mask = targets == 0

        pos_predictions = self.sigmoid(links[pos_mask])
        neg_predictions = self.sigmoid(links[neg_mask])
        return {
            **self.compute_loss(links=links, targets=targets, pos_mask=pos_mask, neg_mask=neg_mask),
            **self.compute_accuracy(pos_predictions=pos_predictions, neg_predictions=neg_predictions),
            **self.compute_hits(pos_predictions=pos_predictions, neg_predictions=neg_predictions),
        }

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        metrics = self.evaluate_metrics(outputs=outputs)

        self.log("pos_loss", {"train": metrics["pos_loss"]}, on_epoch=True)
        self.log("neg_loss", {"train": metrics["neg_loss"]}, on_epoch=True)
        self.log("loss", {"train": metrics["total_loss"]}, on_epoch=True)

        self.log("pos_acc", {"train": metrics["pos_acc"]}, on_epoch=True)
        self.log("neg_acc", {"train": metrics["neg_acc"]}, on_epoch=True)
        self.log("accuracy", {"train": metrics["total_acc"]}, on_epoch=True)

        self.log("hits50", {"train": metrics["hits@50"]}, on_epoch=True)

        if self.trainer is not None:
            self.log("step", self.trainer.current_epoch)

    def validation_step(  # type: ignore # pylint: disable=arguments-differ
        # Signature of "training_step" incompatible with supertype "LightningModule"
        # No need for all arguments
        self,
        batch: Tuple[dgl.DGLHeteroGraph, torch.Tensor],
        _: int,
    ) -> Optional[STEP_OUTPUT]:
        return self.training_step(batch=batch, _=_)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        metrics = self.evaluate_metrics(outputs=outputs)
        self.log("validation_loss", metrics["total_loss"], on_epoch=True)
        self.log("validation_hits", metrics["hits@50"], on_epoch=True)

        self.log("pos_loss", {"valid": metrics["pos_loss"]}, on_epoch=True)
        self.log("neg_loss", {"valid": metrics["neg_loss"]}, on_epoch=True)
        self.log("loss", {"valid": metrics["total_loss"]}, on_epoch=True)

        self.log("pos_acc", {"valid": metrics["pos_acc"]}, on_epoch=True)
        self.log("neg_acc", {"valid": metrics["neg_acc"]}, on_epoch=True)
        self.log("accuracy", {"valid": metrics["total_acc"]}, on_epoch=True)

        self.log("hits50", {"valid": metrics["hits@50"]}, on_epoch=True)

        if self.trainer is not None:
            self.log("step", self.trainer.current_epoch)

    def test_step(  # type: ignore # pylint: disable=arguments-differ
        # Signature of "training_step" incompatible with supertype "LightningModule"
        # No need for all arguments
        self,
        batch: Tuple[dgl.DGLHeteroGraph, torch.Tensor],
        _: int,
    ) -> Optional[STEP_OUTPUT]:
        return self.training_step(batch=batch, _=_)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        metrics = self.evaluate_metrics(outputs=outputs)

        self.log("test_pos_loss", metrics["pos_loss"], on_epoch=True)
        self.log("test_neg_loss", metrics["neg_loss"], on_epoch=True)
        self.log("test_loss", metrics["total_loss"], on_epoch=True)

        self.log("test_pos_acc", metrics["pos_acc"], on_epoch=True)
        self.log("test_neg_acc", metrics["neg_acc"], on_epoch=True)
        self.log("test_accuracy", metrics["total_acc"], on_epoch=True)

        self.log("test_hits50", metrics["hits@50"], on_epoch=True)

    def predict_step(
        self, batch: Tuple[dgl.DGLHeteroGraph, torch.Tensor], _: int, __: Optional[int] = None
    ) -> torch.Tensor:
        graph, ___ = batch
        predicted_links = self(graph)
        return self.sigmoid(predicted_links.detach().cpu()) >= 0.5
