"""
Link Prediction PyTorch Lightning Module
"""
from typing import Dict, Optional

import dgl
import pytorch_lightning as pl
import torch.optim
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import nn

from zero_to_hero.models.models import MLP, GraphEncoder


class LinkPredictorDDI(pl.LightningModule):  # pylint: disable=too-many-ancestors
    # the ancestors come from PyTorch Lightning
    """
    Implementation of the Link Predictor LightModule
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()

        self.config = config

        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

        self.graph_encoder = GraphEncoder(
            in_features=self.config["model"]["in_features"],
            list_out_features=self.config["model"]["out_features"],
            list_gnn_dropout=self.config["model"]["gnn_dropout"],
        )
        self.link_predictor = MLP(
            n_targets=1,
            n_features=self.config["model"]["out_features"][-1],
            hidden_nodes=self.config["model"]["hidden_nodes"],
        )

        self.embeds = torch.empty(0)

    def forward(  # type: ignore # pylint: disable=arguments-differ
        # Signature of "forward" incompatible with supertype "LightningModule"
        # No need for all arguments
        self,
        feats: torch.Tensor,
        blocks: dgl.DGLHeteroGraph,
    ) -> torch.Tensor:
        self.embeds = self.graph_encoder(feats, blocks)
        return self.link_predictor(self.embeds)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        pass

    def training_step_end(self, *args, **kwargs) -> STEP_OUTPUT:
        pass

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        pass

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        pass

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        pass

    def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        pass

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        pass
