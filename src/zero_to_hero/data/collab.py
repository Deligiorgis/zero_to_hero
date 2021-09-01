"""
DataModule for the OGB Drug-Drug-Interaction dataset
DataModule for link-prediction
"""
from typing import Dict, List, Optional

import dgl
import numpy as np
import pytorch_lightning as pl
import torch
from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.dataloading.dataloader import EdgeCollator, NodeCollator
from ogb.linkproppred import DglLinkPropPredDataset
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset

from zero_to_hero.data.statistics import standardize_data


class EdgeDataset(Dataset):
    """
    Custom PyTorch Dataset that provides the evaluation edges
    """

    def __init__(self, edges: torch.Tensor) -> None:
        self.edges = edges if edges.shape[1] == 2 else edges.T

    def __len__(self) -> int:
        return self.edges.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.edges[idx]


class CollabDataModule(pl.LightningDataModule):
    """
    Implementation of the PyTorch Lightning DataModule for the OGBL dataset Drug-Drug-Interaction
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()

        self.config = config

        self.train_edge_collator: Optional[EdgeCollator] = None
        self.valid_node_collator: Optional[List[NodeCollator]] = None
        self.test_node_collator: Optional[List[NodeCollator]] = None

        self.splits: Optional[Dict[str, Dict[str, torch.Tensor]]] = None

        np.random.seed(1)

    def prepare_data(self) -> None:
        DglLinkPropPredDataset(
            name="ogbl-collab",
            root="data",
        )

    @staticmethod
    def get_graph_per_year(
        graph: dgl.DGLHeteroGraph,
        years: torch.Tensor,
        add_self_loop_per_year: bool = True,
    ) -> List[dgl.DGLHeteroGraph]:
        """

        :param graph:
        :param years:
        :param add_self_loop_per_year:
        :return:
        """
        eids_mask_per_year = graph.edata["year"] == years
        graph_per_year = [
            dgl.edge_subgraph(graph=graph, edges=eids, relabel_nodes=False) for eids in eids_mask_per_year.T
        ]
        if add_self_loop_per_year:
            graph_with_self_loop_per_year = list(map(dgl.add_self_loop, graph_per_year))
            return graph_with_self_loop_per_year
        return graph_per_year

    @staticmethod
    def convert_to_single_graph(multi_graph: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        """

        :param multi_graph:
        :return:
        """
        edges = torch.vstack(multi_graph.edges(form="uv")).unique(dim=1)
        single_graph = dgl.graph((edges[0], edges[1]), num_nodes=multi_graph.num_nodes())
        single_graph.srcdata["feat"] = multi_graph.srcdata["feat"]
        return single_graph

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = DglLinkPropPredDataset(
            name="ogbl-collab",
            root="data",
        )
        graph: dgl.DGLHeteroGraph = dataset.graph[0]
        self.splits = dataset.get_edge_split()

        # Convert to single graph
        graph = self.convert_to_single_graph(graph)
        non_diagonal_eids = torch.arange(graph.num_edges())

        # Standardize features
        graph.srcdata["feat"] = torch.from_numpy(
            standardize_data(
                data=graph.srcdata["feat"].numpy(),
                return_moments=False,
            )
        )

        graph = dgl.add_self_loop(graph)

        node_ids = torch.arange(graph.num_nodes())
        eval_block_sampler = MultiLayerFullNeighborSampler(1)

        num_layers = len(self.config["model"]["graph"]["out_features"]) + 1

        if stage in ("fit", "validation", None):
            train_block_sampler = MultiLayerFullNeighborSampler(num_layers)
            negative_sampler = dgl.dataloading.negative_sampler.Uniform(1)

            self.train_edge_collator = EdgeCollator(
                g=graph,
                eids=non_diagonal_eids,
                block_sampler=train_block_sampler,
                negative_sampler=negative_sampler,
            )

            self.valid_node_collator = [
                NodeCollator(
                    g=graph,
                    nids=node_ids,
                    block_sampler=eval_block_sampler,
                )
                for _ in range(num_layers - 1)
            ]

        if stage in ("test", None):
            graph.add_edges(
                u=self.splits["valid"]["edge"][:, 0],
                v=self.splits["valid"]["edge"][:, 1],
            )
            graph = self.convert_to_single_graph(graph)
            self.test_node_collator = [
                NodeCollator(
                    g=graph,
                    nids=node_ids,
                    block_sampler=eval_block_sampler,
                )
                for _ in range(num_layers - 1)
            ]

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
        return [
            DataLoader(
                dataset=collator.dataset,
                collate_fn=collator.collate,
                batch_size=self.config["hyper_parameters"]["eval_batch_size"],
                shuffle=False,
                pin_memory=self.config["hyper_parameters"]["pin_memory"],
                num_workers=self.config["hyper_parameters"]["num_workers"],
                drop_last=False,
            )
            for collator in self.valid_node_collator
        ] + [
            DataLoader(
                dataset=EdgeDataset(self.splits["valid"]["edge"]),
                batch_size=self.config["hyper_parameters"]["eval_batch_size"],
                shuffle=False,
                pin_memory=self.config["hyper_parameters"]["pin_memory"],
                num_workers=self.config["hyper_parameters"]["num_workers"],
                drop_last=False,
            ),
            DataLoader(
                dataset=EdgeDataset(self.splits["valid"]["edge_neg"]),
                batch_size=self.config["hyper_parameters"]["eval_batch_size"],
                shuffle=False,
                pin_memory=self.config["hyper_parameters"]["pin_memory"],
                num_workers=self.config["hyper_parameters"]["num_workers"],
                drop_last=False,
            ),
        ]

    def test_dataloader(self) -> EVAL_DATALOADERS:
        assert self.test_node_collator is not None
        return [
            DataLoader(
                dataset=collator.dataset,
                collate_fn=collator.collate,
                batch_size=self.config["hyper_parameters"]["eval_batch_size"],
                shuffle=False,
                pin_memory=self.config["hyper_parameters"]["pin_memory"],
                num_workers=self.config["hyper_parameters"]["num_workers"],
                drop_last=False,
            )
            for collator in self.test_node_collator
        ] + [
            DataLoader(
                dataset=EdgeDataset(self.splits["test"]["edge"]),
                batch_size=self.config["hyper_parameters"]["eval_batch_size"],
                shuffle=False,
                pin_memory=self.config["hyper_parameters"]["pin_memory"],
                num_workers=self.config["hyper_parameters"]["num_workers"],
                drop_last=False,
            ),
            DataLoader(
                dataset=EdgeDataset(self.splits["test"]["edge_neg"]),
                batch_size=self.config["hyper_parameters"]["eval_batch_size"],
                shuffle=False,
                pin_memory=self.config["hyper_parameters"]["pin_memory"],
                num_workers=self.config["hyper_parameters"]["num_workers"],
                drop_last=False,
            ),
        ]
