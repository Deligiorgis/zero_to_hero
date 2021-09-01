"""
DataModule for the OGB Drug-Drug-Interaction dataset
DataModule for link-prediction
"""
from typing import Dict, Optional

import dgl
import pytorch_lightning as pl
import torch
from dgl.dataloading import MultiLayerFullNeighborSampler, MultiLayerNeighborSampler
from dgl.dataloading.dataloader import EdgeCollator, NodeCollator
from ogb.linkproppred import DglLinkPropPredDataset
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader


class OGBLDrugDrugInteractionDataModule(pl.LightningDataModule):
    """
    Implementation of the PyTorch Lightning DataModule for the OGBL dataset Drug-Drug-Interaction
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()

        self.config = config

        self.train_edge_collator: Optional[EdgeCollator] = None
        self.valid_node_collator: Optional[NodeCollator] = None
        self.test_node_collator: Optional[NodeCollator] = None

    def prepare_data(self) -> None:
        DglLinkPropPredDataset(
            name="ogbl-ddi",
            root="data",
        )

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = DglLinkPropPredDataset(
            name="ogbl-ddi",
            root="data",
        )
        graph = dataset.graph[0]
        splits = dataset.get_edge_split()

        eval_block_sampler = MultiLayerFullNeighborSampler(1)

        graph = dgl.add_self_loop(graph)
        node_ids = torch.arange(graph.num_nodes())

        if stage in ("fit", "validation", None):
            train_block_sampler = MultiLayerNeighborSampler(self.config[""])
            self.train_edge_collator = EdgeCollator(
                g=graph,
                eids=graph.edge_ids(splits["train"]["edge"][:, 0], splits["train"]["edge"][:, 1]),
                block_sampler=train_block_sampler,
            )

            self.valid_node_collator = NodeCollator(
                g=graph,
                nids=node_ids,
                block_sampler=eval_block_sampler,
            )

        if stage in ("test", None):
            self.test_node_collator = NodeCollator(
                g=graph,
                nids=node_ids,
                block_sampler=eval_block_sampler,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        assert self.train_edge_collator is not None
        return DataLoader(
            dataset=self.train_edge_collator.dataset,
            collate_fn=self.train_edge_collator.collate,
            batch_size=self.config["hyper_parameters"]["train_batch_size"],
            shuffle=True,
            pin_memory=self.config["hyper_parameters"]["pin_memory"],
            num_workers=self.config["hyper_parameters"]["num_workers"],
            drop_last=False,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        assert self.valid_node_collator is not None
        return DataLoader(
            dataset=self.valid_node_collator.dataset,
            collate_fn=self.valid_node_collator.collate,
            batch_size=self.config["hyper_parameters"]["eval_batch_size"],
            shuffle=False,
            pin_memory=self.config["hyper_parameters"]["pin_memory"],
            num_workers=self.config["hyper_parameters"]["num_workers"],
            drop_last=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        assert self.test_node_collator is not None
        return DataLoader(
            dataset=self.test_node_collator.dataset,
            collate_fn=self.test_node_collator.collate,
            batch_size=self.config["hyper_parameters"]["eval_batch_size"],
            shuffle=False,
            pin_memory=self.config["hyper_parameters"]["pin_memory"],
            num_workers=self.config["hyper_parameters"]["num_workers"],
            drop_last=False,
        )
