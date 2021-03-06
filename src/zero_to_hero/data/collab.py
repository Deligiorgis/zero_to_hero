"""
DataModule for the OGB Drug-Drug-Interaction dataset
DataModule for link-prediction
"""
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import dgl
import numpy as np
import pytorch_lightning as pl
import torch
from dgl.data.graph_serialize import load_labels_v2
from dgl.dataloading.negative_sampler import Uniform
from ogb.linkproppred import DglLinkPropPredDataset
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from scipy.sparse.csgraph import shortest_path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from zero_to_hero.data.statistics import standardize_data


class GraphDataSet(Dataset):
    """
    GraphDataset for torch DataLoader
    """

    def __init__(self, path: Union[Path, str]) -> None:
        self.path = str(path)
        self.links = load_labels_v2(filename=self.path)["links"]

    def __len__(self) -> int:
        return self.links.shape[0]

    def __getitem__(self, index: int) -> Tuple[dgl.DGLHeteroGraph, torch.Tensor]:
        graph, _ = dgl.load_graphs(filename=self.path, idx_list=[index])
        return graph[0], self.links[index]


class EdgeDataSet(Dataset):
    """
    Edge Dataset for speeding up the sampling of the graphs
    """

    def __init__(
        self,
        edges: torch.Tensor,
        links: torch.Tensor,
        transform: Callable[[torch.Tensor], dgl.DGLHeteroGraph],
    ) -> None:
        self.edges = edges
        self.transform = transform
        self.links = links

    def __len__(self) -> int:
        return self.edges.shape[0]

    def __getitem__(self, index: int) -> Tuple[dgl.DGLHeteroGraph, torch.Tensor]:
        subgraph = self.transform(self.edges[index])
        return subgraph, self.links[index]


def double_radius_node_labeling(subgraph: dgl.DGLHeteroGraph, src: int, dst: int) -> torch.Tensor:
    """
    Double Radius Node labeling
    d = r(i, u) + r(i, v)
    node_label = 1 + min(r(i, u), r(i, v)) + (d // 2) * (d // 2 + d % 2 - 1)
    Isolated nodes in subgraph will be set as zero.
    Extreme large graph may cause memory error.
    Args:
        subgraph(DGLGraph): The graph
        src(int): node id of one of src node in new subgraph
        dst(int): node id of one of dst node in new subgraph
    Returns:
        node_label(Tensor): node labeling tensor
    """
    adj = subgraph.adj().to_dense().numpy()
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    node_label = 1 + torch.min(dist2src, dist2dst)
    node_label += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    node_label[src] = 1.0
    node_label[dst] = 1.0
    node_label[torch.isnan(node_label)] = 0.0

    return node_label.to(torch.long)


class CollabDataModule(pl.LightningDataModule):
    """
    Implementation of the PyTorch Lightning DataModule for the OGBL dataset Drug-Drug-Interaction
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()

        self.config = config

        self.dataset: Dict[str, Optional[GraphDataSet]] = {
            "train": None,
            "valid": None,
            "test": None,
            "predict": None,
        }

        self.ndata: torch.Tensor = torch.empty(0)

        self.edata: Dict[str, torch.Tensor] = {
            "fit": torch.empty(0),
            "test": torch.empty(0),
            "predict": torch.empty(0),
        }

    def prepare_data(self) -> None:
        DglLinkPropPredDataset(
            name="ogbl-collab",
            root="data",
        )

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = DglLinkPropPredDataset(
            name="ogbl-collab",
            root="data",
        )
        multi_graph: Dict[str, dgl.DGLHeteroGraph] = {"fit": dataset[0]}
        split_edge = dataset.get_edge_split()

        multi_graph["fit"] = dgl.add_self_loop(g=multi_graph["fit"])

        for key, values in multi_graph["fit"].edata.items():
            multi_graph["fit"].edata[key] = values.float()

        multi_graph["test"] = deepcopy(multi_graph["fit"])
        if self.config["data"]["test_graph_with_valid_edges"]:
            multi_graph["test"] = dgl.add_edges(
                g=multi_graph["fit"],
                u=split_edge["valid"]["edge"][:, 0],
                v=split_edge["valid"]["edge"][:, 1],
                data={
                    "weight": split_edge["valid"]["weight"].unsqueeze(1).float(),
                    "year": split_edge["valid"]["year"].unsqueeze(1).float(),
                },
            )

        multi_graph["predict"] = deepcopy(multi_graph["test"])
        if self.config["data"]["predict_graph_with_test_edges"]:
            multi_graph["predict"] = dgl.add_edges(
                g=multi_graph["test"],
                u=split_edge["test"]["edge"][:, 0],
                v=split_edge["test"]["edge"][:, 1],
                data={
                    "weight": split_edge["test"]["weight"].unsqueeze(1).float(),
                    "year": split_edge["test"]["year"].unsqueeze(1).float(),
                },
            )

        simple_graph = {
            phase: dgl.to_simple(
                g=multi_graph[phase],
                copy_ndata=True,
                copy_edata=True,
                aggregator="sum",
            )
            for phase in ["fit", "test", "predict"]
        }

        ndata = simple_graph["fit"].ndata["feat"]
        if self.config["data"]["standardize_data"]:
            ndata = torch.from_numpy(
                standardize_data(
                    data=ndata.numpy(),
                    return_moments=False,
                )
            )
        self.ndata = ndata

        self.edata = {phase: simple_graph[phase].edata["weight"].float() for phase in ["fit", "test", "predict"]}

        for phase in ["fit", "test", "predict"]:
            simple_graph[phase].ndata.clear()
            simple_graph[phase].edata.clear()

        if stage in ("train", "fit", None):
            # Train
            path = (
                Path("data/ogbl_collab_seal")
                / f"train_{self.config['data']['hop']}-hop_{self.config['data']['subsample_train_ratio']}-subsample.bin"
            )
            if not path.exists():
                edges, links = self.generate_edges_and_links(
                    split_edge=split_edge,
                    graph=simple_graph["fit"],
                    phase="train",
                )
                graph_list, links = self.generate_list_of_graphs_and_links(
                    edges=edges,
                    links=links,
                    graph=simple_graph["fit"],
                )
                dgl.save_graphs(str(path), graph_list, {"links": links})
            self.dataset["train"] = GraphDataSet(path=path)

            # Valid
            path = (
                Path("data/ogbl_collab_seal") / f"valid_{self.config['data']['hop']}-hop_"
                f"{self.config['data']['subsample_test_ratio']}-subsample.bin"
            )
            if not path.exists():
                edges, links = self.generate_edges_and_links(
                    split_edge=split_edge,
                    graph=simple_graph["fit"],
                    phase="valid",
                )
                graph_list, links = self.generate_list_of_graphs_and_links(
                    edges=edges,
                    links=links,
                    graph=simple_graph["fit"],
                )
                dgl.save_graphs(str(path), graph_list, {"links": links})
            self.dataset["valid"] = GraphDataSet(path=path)

        if stage in ("test", None):
            path = (
                Path("data/ogbl_collab_seal") / f"test_{self.config['data']['hop']}-hop_"
                f"{'with' if self.config['data']['test_graph_with_valid_edges'] else 'without'}-valid-edges_"
                f"{self.config['data']['subsample_test_ratio']}-subsample.bin"
            )
            if not path.exists():
                edges, links = self.generate_edges_and_links(
                    split_edge=split_edge,
                    graph=simple_graph["test"],
                    phase="test",
                )
                graph_list, links = self.generate_list_of_graphs_and_links(
                    edges=edges,
                    links=links,
                    graph=simple_graph["test"],
                )
                dgl.save_graphs(str(path), graph_list, {"links": links})
            self.dataset["test"] = GraphDataSet(path=path)

        if stage in ("predict", None):
            path = (
                Path("data/ogbl_collab_seal") / f"predict_{self.config['data']['hop']}-hop_"
                f"{'with' if self.config['data']['test_graph_with_valid_edges'] else 'without'}-valid-edges_"
                f"{'with' if self.config['data']['predict_graph_with_test_edges'] else 'without'}-test-edges_"
                f"{self.config['data']['subsample_test_ratio']}-subsample.bin"
            )
            split_edge["predict"] = {
                "edge": torch.randperm(100).view(-1, 2).float(),  # Random collaborations
                "edge_neg": torch.empty(0),
            }
            if not path.exists():
                edges, links = self.generate_edges_and_links(
                    split_edge=split_edge,
                    graph=simple_graph["predict"],
                    phase="predict",
                )
                graph_list, links = self.generate_list_of_graphs_and_links(
                    edges=edges,
                    links=links,
                    graph=simple_graph["predict"],
                )
                dgl.save_graphs(str(path), graph_list, {"links": links})
            self.dataset["predict"] = GraphDataSet(path=path)

    def sample_subgraph(self, target_nodes: torch.Tensor, graph: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        """

        Parameters
        ----------
        target_nodes
        graph

        Returns
        -------

        """
        list_sample_nodes = [target_nodes]
        frontiers = target_nodes

        for _ in range(self.config["data"]["hop"]):
            frontiers = graph.out_edges(frontiers)[1]
            frontiers = torch.unique(frontiers)
            list_sample_nodes.append(frontiers)

        sample_nodes = torch.cat(list_sample_nodes)
        sample_nodes = torch.unique(sample_nodes)
        subgraph = dgl.node_subgraph(graph, sample_nodes)

        u_id = int(torch.nonzero(subgraph.ndata[dgl.NID] == int(target_nodes[0]), as_tuple=False))
        v_id = int(torch.nonzero(subgraph.ndata[dgl.NID] == int(target_nodes[1]), as_tuple=False))

        if subgraph.has_edges_between(u_id, v_id):
            link_id = subgraph.edge_ids(u_id, v_id, return_uv=True)[2]
            subgraph.remove_edges(link_id)
        if subgraph.has_edges_between(v_id, u_id):
            link_id = subgraph.edge_ids(v_id, u_id, return_uv=True)[2]
            subgraph.remove_edges(link_id)

        subgraph.ndata["label"] = double_radius_node_labeling(subgraph, u_id, v_id)
        return subgraph

    @staticmethod
    def _collate(batch: List[Tuple[dgl.DGLHeteroGraph, torch.Tensor]]) -> Tuple[dgl.DGLHeteroGraph, torch.Tensor]:
        batch_graphs, batch_links = tuple(map(list, zip(*batch)))
        return dgl.batch(batch_graphs), torch.stack(batch_links)  # type: ignore # False Positive from MyPy
        # batch_links: List[torch.Tensor]

    def generate_list_of_graphs_and_links(
        self,
        edges: torch.Tensor,
        links: torch.Tensor,
        graph: dgl.DGLHeteroGraph,
    ) -> Tuple[List[dgl.DGLHeteroGraph], torch.Tensor]:
        """

        Parameters
        ----------
        edges
        links
        graph

        Returns
        -------

        """
        sample_subgraph_partial = partial(self.sample_subgraph, graph=graph)
        edge_dataset = EdgeDataSet(
            edges=edges,
            links=links,
            transform=sample_subgraph_partial,
        )
        sampler = DataLoader(
            edge_dataset,
            batch_size=self.config["data"]["sampler"]["batch_size"],
            num_workers=self.config["data"]["sampler"]["num_workers"],
            pin_memory=self.config["data"]["sampler"]["pin_memory"],
            shuffle=False,
            collate_fn=self._collate,
        )

        subgraph_list = []
        links_list = []
        for subgraph, sub_links in tqdm(sampler, ncols=100):
            label_copy = deepcopy(sub_links)
            subgraph = dgl.unbatch(subgraph)

            del sub_links
            subgraph_list += subgraph
            links_list.append(label_copy)

        return subgraph_list, torch.cat(links_list)

    def generate_edges_and_links(
        self,
        split_edge: Dict[str, Dict[str, torch.Tensor]],
        graph: dgl.DGLHeteroGraph,
        phase: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Parameters
        ----------
        split_edge
        graph
        phase

        Returns
        -------

        """
        pos_edges = split_edge[phase]["edge"]
        if phase == "train":
            neg_sampler = Uniform(k=1)
            eids = graph.edge_ids(u=pos_edges[:, 0], v=pos_edges[:, 1])
            neg_edges = torch.stack(neg_sampler(graph, eids), dim=1)
        else:
            neg_edges = split_edge[phase]["edge_neg"]

        pos_edges = self.subsample_edges(
            edges=pos_edges,
            subsample_ratio=self.config["data"]["subsample_train_ratio"]
            if phase == "train"
            else self.config["data"]["subsample_test_ratio"],
        ).long()
        neg_edges = self.subsample_edges(
            edges=neg_edges,
            subsample_ratio=self.config["data"]["subsample_train_ratio"]
            if phase == "train"
            else self.config["data"]["subsample_test_ratio"],
        ).long()

        edges = torch.cat([pos_edges, neg_edges])
        links = torch.cat([torch.ones(pos_edges.shape[0], 1), torch.zeros(neg_edges.shape[0], 1)])
        edges, links = self.shuffle_edges_and_links(edges=edges, links=links)

        return edges, links

    @staticmethod
    def subsample_edges(edges: torch.Tensor, subsample_ratio: float) -> torch.Tensor:
        """

        :param edges:
        :param subsample_ratio:
        :return:
        """
        num_edges = edges.shape[0]
        perm = torch.randperm(num_edges)
        perm = perm[: int(subsample_ratio * num_edges)]
        edges = edges[perm]
        return edges

    @staticmethod
    def shuffle_edges_and_links(edges: torch.Tensor, links: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Parameters
        ----------
        edges
        links

        Returns
        -------

        """
        perm = torch.randperm(edges.shape[0])
        edges = edges[perm]
        links = links[perm]
        return edges, links

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        assert self.dataset["train"] is not None
        return DataLoader(
            dataset=self.dataset["train"],
            batch_size=self.config["data"]["batch_size"],
            shuffle=True,
            num_workers=self.config["data"]["num_workers"],
            collate_fn=dgl.dataloading.GraphCollator().collate,
            pin_memory=self.config["data"]["pin_memory"],
            drop_last=False,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        assert self.dataset["valid"] is not None
        return DataLoader(
            dataset=self.dataset["valid"],
            batch_size=self.config["data"]["batch_size"],
            shuffle=False,
            num_workers=self.config["data"]["num_workers"],
            collate_fn=dgl.dataloading.GraphCollator().collate,
            pin_memory=self.config["data"]["pin_memory"],
            drop_last=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        assert self.dataset["test"] is not None
        return DataLoader(
            dataset=self.dataset["test"],
            batch_size=self.config["data"]["batch_size"],
            shuffle=False,
            num_workers=self.config["data"]["num_workers"],
            collate_fn=dgl.dataloading.GraphCollator().collate,
            pin_memory=self.config["data"]["pin_memory"],
            drop_last=False,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        assert self.dataset["predict"] is not None
        return DataLoader(
            dataset=self.dataset["predict"],
            batch_size=self.config["data"]["batch_size"],
            shuffle=False,
            num_workers=self.config["data"]["num_workers"],
            collate_fn=dgl.dataloading.GraphCollator().collate,
            pin_memory=self.config["data"]["pin_memory"],
            drop_last=False,
        )
