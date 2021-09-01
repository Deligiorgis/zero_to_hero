"""
Link Prediction PyTorch Lightning Module
"""
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import dgl
import pytorch_lightning as pl
import torch.optim
from dgl.heterograph import DGLBlock
from ogb.linkproppred import Evaluator
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import nn

from zero_to_hero.models.models import MLP, GraphEncoder


class LinkPredictorCollab(pl.LightningModule):  # pylint: disable=too-many-ancestors
    # the ancestors come from PyTorch Lightning
    """
    Implementation of the Link Predictor LightModule
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()

        self.config = config

        self.criterion = nn.BCEWithLogitsLoss(reduction="sum")

        self.graph_encoder = GraphEncoder(
            in_features=self.config["data"]["in_features"],
            list_out_features=self.config["model"]["graph"]["out_features"],
            list_gnn_dropout=self.config["model"]["graph"]["dropout"],
        )
        self.link_predictor = MLP(
            in_features=self.config["model"]["graph"]["out_features"][-1],
            list_out_features=self.config["model"]["linear"]["out_features"],
            list_linear_dropout=self.config["model"]["linear"]["dropout"],
        )

        self.sigmoid = torch.nn.Sigmoid()

        self.embeds = {
            "main": torch.empty(0),
            "input": torch.empty(0),
            "output": torch.empty(0),
        }

        self.last_batch_idx = -1

        self.save_hyperparameters()

    def forward(  # type: ignore # pylint: disable=arguments-differ
        # Signature of "forward" incompatible with supertype "LightningModule"
        # No need for all arguments
        self,
        feats: torch.Tensor,
        blocks: List[DGLBlock],
        pos_data: dgl.DGLHeteroGraph,
        neg_data: dgl.DGLHeteroGraph,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        self.embeds["main"] = self.graph_encoder(feats, blocks)

        pos_nodes = pos_data.edges()
        pos_edges = self.link_predictor(self.embeds["main"][pos_nodes[0]] * self.embeds["main"][pos_nodes[1]])

        neg_nodes = neg_data.edges()
        neg_edges = self.link_predictor(self.embeds["main"][neg_nodes[0]] * self.embeds["main"][neg_nodes[1]])

        return pos_edges, neg_edges

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config["hyper_parameters"]["learning_rate"],
        )

    def compute_link_loss(self, pos_links: torch.Tensor, neg_links: torch.Tensor) -> Dict[str, torch.Tensor]:
        """

        :param pos_links:
        :param neg_links:
        :return:
        """
        pos_loss = self.criterion(pos_links, torch.ones_like(pos_links))
        neg_loss = self.criterion(neg_links, torch.zeros_like(neg_links))
        loss = (pos_loss + neg_loss) / (pos_links.shape[0] + neg_links.shape[0])
        return {
            "loss": loss,
            "pos_loss": pos_loss.item() / pos_links.shape[0],
            "neg_loss": neg_loss.item() / neg_links.shape[0],
        }

    @staticmethod
    @torch.no_grad()
    def compute_accuracy(pos_predictions: torch.Tensor, neg_predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """

        :param pos_predictions:
        :param neg_predictions:
        :return:
        """
        pos_accuracy: torch.Tensor = (pos_predictions >= 0.5).float().sum()
        neg_accuracy: torch.Tensor = (neg_predictions < 0.5).float().sum()
        accuracy: torch.Tensor = (pos_accuracy + neg_accuracy) / (pos_predictions.shape[0] + neg_predictions.shape[0])
        return {
            "accuracy": accuracy,
            "pos_accuracy": pos_accuracy / pos_predictions.shape[0],
            "neg_accuracy": neg_accuracy / neg_predictions.shape[0],
        }

    def training_step(  # type: ignore # pylint: disable=arguments-differ
        # Signature of "forward" incompatible with supertype "LightningModule"
        # No need for all arguments
        self,
        batch: Tuple[
            torch.Tensor,
            dgl.DGLHeteroGraph,
            dgl.DGLHeteroGraph,
            List[DGLBlock],
        ],
        _: int,
    ) -> STEP_OUTPUT:
        __, pos_graph, neg_graph, blocks = batch
        pos_links, neg_links = self(blocks[0].srcdata["feat"], blocks, pos_graph, neg_graph)

        return_dict = self.compute_link_loss(
            pos_links=pos_links,
            neg_links=neg_links,
        )

        return_dict.update(
            {
                "pos_links": pos_links.detach().cpu(),
                "neg_links": neg_links.detach().cpu(),
            }
        )

        return return_dict

    @torch.no_grad()
    def compute_graph_metrics(
        self,
        outputs: EPOCH_OUTPUT,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """

        :param outputs:
        :return:
        """
        pos_links, neg_links = [], []
        for output in outputs:
            assert isinstance(output, dict)
            pos_links.append(output["pos_links"])
            neg_links.append(output["neg_links"])

        all_pos_links = torch.cat(pos_links)
        all_neg_links = torch.cat(neg_links)

        dict_loss = self.compute_link_loss(
            pos_links=all_pos_links,
            neg_links=all_neg_links,
        )

        pos_predictions = self.sigmoid(all_pos_links)
        neg_predictions = self.sigmoid(all_neg_links)

        dict_acc = self.compute_accuracy(
            pos_predictions=pos_predictions,
            neg_predictions=neg_predictions,
        )

        evaluator = Evaluator(name="ogbl-collab")
        hits = evaluator.eval(
            {
                "y_pred_pos": pos_predictions.squeeze(),
                "y_pred_neg": neg_predictions.squeeze(),
            }
        )

        return dict_loss, dict_acc, hits

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        dict_loss, dict_acc, hits = self.compute_graph_metrics(outputs=outputs)

        self.log("loss", {"train": dict_loss["loss"]}, on_epoch=True)
        self.log("pos_loss", {"train": dict_loss["pos_loss"]}, on_epoch=True)
        self.log("neg_loss", {"train": dict_loss["neg_loss"]}, on_epoch=True)

        self.log("accuracy", {"train": dict_acc["accuracy"]}, on_epoch=True)
        self.log("pos_accuracy", {"train": dict_acc["pos_accuracy"]}, on_epoch=True)
        self.log("neg_accuracy", {"train": dict_acc["neg_accuracy"]}, on_epoch=True)

        self.log("Hits50", {"train": hits["hits@50"]}, on_epoch=True)

        if self.trainer is not None:
            self.log("step", self.trainer.current_epoch)

    def update_eval_embeds(
        self,
        batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, List[dgl.DGLHeteroGraph]]],
        embeds: torch.Tensor,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """

        :param batch:
        :param embeds:
        :param batch_idx:
        :param dataloader_idx:
        :return:
        """
        _, output_nodes, blocks = batch
        if batch_idx == 0:
            self.embeds["output"] = torch.zeros(blocks[0].num_nodes(), embeds.shape[-1])
        self.embeds["output"][output_nodes] = embeds

        if dataloader_idx == 0:
            self.last_batch_idx = batch_idx
            self.embeds["input"] = deepcopy(self.embeds["output"])
        else:
            if batch_idx == self.last_batch_idx:
                self.embeds["input"] = deepcopy(self.embeds["output"])

    def get_eval_embeds(
        self,
        batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, List[dgl.DGLHeteroGraph]]],
        dataloader_idx: int,
    ) -> torch.Tensor:
        """

        :param batch:
        :param dataloader_idx:
        :return:
        """
        input_nodes, _, blocks = batch
        if dataloader_idx == 0:
            feat = blocks[0].srcdata["feat"]
        else:
            feat = self.embeds["input"][input_nodes]

        embeds = self.graph_encoder.layers[2 * dataloader_idx](
            graph=blocks[0],
            feat=feat,
        )
        if dataloader_idx + 1 < len(self.config["model"]["graph"]["out_features"]):
            embeds = self.graph_encoder.layers[2 * dataloader_idx + 1](embeds)
            return embeds
        return embeds

    def validation_step(  # type: ignore # pylint: disable=arguments-differ
        # Signature of "training_step" incompatible with supertype "LightningModule"
        # No need for all arguments
        self,
        batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, List[dgl.DGLHeteroGraph]]],
        batch_idx: int,
        dataloader_idx: int,
    ) -> Optional[STEP_OUTPUT]:
        num_layers = len(self.config["model"]["graph"]["out_features"])
        if dataloader_idx >= num_layers:
            if batch_idx == 0 and dataloader_idx == num_layers:
                self.embeds["main"] = deepcopy(self.embeds["output"])

            assert isinstance(batch, torch.Tensor)
            links = self.link_predictor(self.embeds["main"][batch[:, 0]] * self.embeds["main"][batch[:, 1]])

            if dataloader_idx == num_layers:
                return {"pos_links": links}
            return {"neg_links": links}

        embeds = self.get_eval_embeds(
            batch=batch,
            dataloader_idx=dataloader_idx,
        )
        self.update_eval_embeds(
            batch=batch,
            embeds=embeds,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )
        return None

    @torch.no_grad()
    def compute_eval_graph_metrics(
        self,
        outputs: EPOCH_OUTPUT,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """

        :param outputs:
        :return:
        """
        pos_links, neg_links = [], []
        for output in outputs:
            if len(output) == 0:
                continue
            for links in output:
                assert isinstance(links, dict)
                if "pos_links" in links:
                    pos_links.append(links["pos_links"])
                else:
                    neg_links.append(links["neg_links"])

        all_pos_links = torch.cat(pos_links)
        all_neg_links = torch.cat(neg_links)

        dict_loss = self.compute_link_loss(
            pos_links=all_pos_links,
            neg_links=all_neg_links,
        )

        pos_predictions = self.sigmoid(all_pos_links)
        neg_predictions = self.sigmoid(all_neg_links)

        dict_acc = self.compute_accuracy(
            pos_predictions=pos_predictions,
            neg_predictions=neg_predictions,
        )

        evaluator = Evaluator(name="ogbl-collab")
        hits = evaluator.eval(
            {
                "y_pred_pos": pos_predictions.squeeze(),
                "y_pred_neg": neg_predictions.squeeze(),
            }
        )

        return dict_loss, dict_acc, hits

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        dict_loss, dict_acc, hits = self.compute_eval_graph_metrics(outputs=outputs)

        if self.trainer is not None:
            self.log("step", self.trainer.current_epoch)

        self.log("validation_loss", dict_loss["loss"], on_epoch=True)

        self.log("loss", {"valid": dict_loss["loss"]}, on_epoch=True)
        self.log("pos_loss", {"valid": dict_loss["pos_loss"]}, on_epoch=True)
        self.log("neg_loss", {"valid": dict_loss["neg_loss"]}, on_epoch=True)

        self.log("accuracy", {"valid": dict_acc["accuracy"]}, on_epoch=True)
        self.log("pos_accuracy", {"valid": dict_acc["pos_accuracy"]}, on_epoch=True)
        self.log("neg_accuracy", {"valid": dict_acc["neg_accuracy"]}, on_epoch=True)

        self.log("Hits50", {"valid": hits["hits@50"]}, on_epoch=True)

    def test_step(  # type: ignore # pylint: disable=arguments-differ
        # Signature of "training_step" incompatible with supertype "LightningModule"
        # No need for all arguments
        self,
        batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, List[dgl.DGLHeteroGraph]]],
        batch_idx: int,
        dataloader_idx: int,
    ) -> Optional[STEP_OUTPUT]:
        return self.validation_step(
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        dict_loss, dict_acc, hits = self.compute_eval_graph_metrics(outputs=outputs)

        self.log("test_loss", dict_loss["loss"], on_epoch=True)
        self.log("test_pos_loss", dict_loss["pos_loss"], on_epoch=True)
        self.log("test_neg_loss", dict_loss["neg_loss"], on_epoch=True)

        self.log("test_accuracy", dict_acc["accuracy"], on_epoch=True)
        self.log("test_pos_accuracy", dict_acc["pos_accuracy"], on_epoch=True)
        self.log("test_neg_accuracy", dict_acc["neg_accuracy"], on_epoch=True)

        self.log("test_Hits50", hits["hits@50"], on_epoch=True)
