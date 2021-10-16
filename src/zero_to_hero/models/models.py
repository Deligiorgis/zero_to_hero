"""
Deep Learning Models that can be re-used in different use-cases
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import dgl
import torch
from dgl.nn.pytorch import GraphConv, SortPooling
from torch import nn


class MLP(nn.Module):
    """
    Implementation of the Multilayer Perceptron model
    """

    def __init__(self, in_features: int, list_out_features: List[int], list_linear_dropout: List[float]) -> None:
        super().__init__()

        linear_layers = []
        for enum, (out_features, dropout) in enumerate(zip(list_out_features, list_linear_dropout)):
            linear_layers.extend(
                [
                    nn.Linear(
                        in_features=in_features if enum == 0 else list_out_features[enum - 1],
                        out_features=out_features,
                        bias=True,
                    ),
                    nn.BatchNorm1d(num_features=out_features),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                ]
            )
        linear_layers = linear_layers[:-3]  # remove last BatchNorm1d, ReLU and Dropout
        self.linear_layers = nn.Sequential(*linear_layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward of the MLP model
        :param data: torch.Tensor
        :return: torch.Tensor
        """
        return self.linear_layers(data)


class CNN(nn.Module):
    """
    Implementation of Convolutional Neural Network Model
    """

    def __init__(  # pylint: disable=too-many-arguments  # Need all parameters for the model
        self,
        in_channels: int,
        list_out_channels: List[int],
        list_kernel_size: List[Union[int, Tuple]],
        list_cnn_dropout: List[float],
        dim: int = 2,
        activation_as_last_layer: bool = False,
    ) -> None:
        super().__init__()

        cnn_layers = []
        for enum, (out_channels, kernel_size, dropout) in enumerate(
            zip(list_out_channels, list_kernel_size, list_cnn_dropout)
        ):
            conv_kwargs = {
                "dim": dim,
                "enum": enum,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "list_out_channels": list_out_channels,
                f"kernel_size_{dim}d": kernel_size,
            }
            cnn_layers.extend(
                [
                    self._get_conv_layer(conv_kwargs=conv_kwargs),
                    self._get_batch_norm_layer(dim=dim, out_channels=out_channels),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                ]
            )
        cnn_layers = cnn_layers[:-3]  # remove last BatchNorm1d/2d, ReLU and Dropout
        if activation_as_last_layer:
            cnn_layers.append(nn.ReLU())
        self.cnn_layers = nn.Sequential(*cnn_layers)

    @staticmethod
    def _get_conv_layer(conv_kwargs: Dict) -> Union[nn.Conv1d, nn.Conv2d]:
        dim = conv_kwargs["dim"]
        if dim == 1:
            kernel_size_1d = conv_kwargs["kernel_size_1d"]
            return nn.Conv1d(
                in_channels=conv_kwargs["in_channels"]
                if conv_kwargs["enum"] == 0
                else conv_kwargs["list_out_channels"][conv_kwargs["enum"] - 1],
                out_channels=conv_kwargs["out_channels"],
                kernel_size=kernel_size_1d,
            )
        if dim == 2:
            kernel_size_2d = conv_kwargs["kernel_size_2d"]
            return nn.Conv2d(
                in_channels=conv_kwargs["in_channels"]
                if conv_kwargs["enum"] == 0
                else conv_kwargs["list_out_channels"][conv_kwargs["enum"] - 1],
                out_channels=conv_kwargs["out_channels"],
                kernel_size=kernel_size_2d,
            )
        raise NotImplementedError("Conv has been implemented only for 1D and 2D.")

    @staticmethod
    def _get_batch_norm_layer(dim: int, out_channels: int) -> Union[nn.BatchNorm1d, nn.BatchNorm2d]:
        if dim == 1:
            return nn.BatchNorm1d(num_features=out_channels)
        if dim == 2:
            return nn.BatchNorm2d(num_features=out_channels)
        raise NotImplementedError("BatchNorm has been implemented only for 1D and 2D.")

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward of the Convolutional Neural Network model
        :return: torch.Tensor
        """
        return self.cnn_layers(data)


class GraphEncoder(nn.Module):
    """
    Graph Encoder using GCN layers
    """

    def __init__(
        self,
        in_features: int,
        list_out_features: List[int],
        list_dropout: Optional[List[float]] = None,
    ) -> None:
        super().__init__()

        self.gcn_layers = nn.ModuleList()
        for enum, out_features in enumerate(list_out_features):
            self.gcn_layers.append(
                GraphConv(
                    in_feats=list_out_features[enum - 1] if enum > 0 else in_features,
                    out_feats=out_features,
                    norm="both",
                    weight=True,
                    allow_zero_in_degree=False,
                )
            )

            if list_dropout is not None:
                self.gcn_layers.append(nn.Dropout(p=list_dropout[enum]))

        if list_dropout is not None:
            self.gcn_layers = self.gcn_layers[:-1]

    def forward(self, graph: dgl.DGLHeteroGraph, node_feats: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        graph
        node_feats
        edge_id

        Returns
        -------

        """
        list_node_feats = [node_feats]
        for layer in self.gcn_layers:
            if isinstance(layer, GraphConv):
                list_node_feats += [
                    torch.tanh(
                        layer(
                            graph=graph,
                            feat=list_node_feats[-1],
                            edge_weight=edge_weight,
                        )
                    )
                ]
            else:
                list_node_feats[-1] = layer(list_node_feats[-1])
        return torch.cat(list_node_feats[1:], dim=-1)


class DGCNN(nn.Module):
    """
    Deep Graph Convolutional Neural Network implementation
    """

    def __init__(
        self,
        ndata: torch.Tensor,
        edata: torch.Tensor,
        config: Dict[str, Any],
    ):
        super().__init__()

        self.node_attributes = ndata
        self.edge_weights = edata

        self.node_embedding = nn.Embedding(
            num_embeddings=ndata.shape[0] + config["model"]["max_z"],
            embedding_dim=config["model"]["embedding_dim"],
        )

        assert (
            len(set(config["model"]["graph_conv"]["out_feats"][:-1]))
            == len(set(config["model"]["graph_conv"]["dropout"][:-1]))
            == 1
        )

        gnn_in_features = self.node_attributes.shape[1] + self.node_embedding.embedding_dim * 2
        self.gnn = GraphEncoder(
            in_features=gnn_in_features,
            list_out_features=config["model"]["graph_conv"]["out_feats"],
            list_dropout=config["model"]["graph_conv"]["dropout"],
        )

        self.pooling = SortPooling(k=config["model"]["sort_pooling"]["k"])

        config["model"]["convolutional"]["kernel_size"][0] = (
            len(config["model"]["graph_conv"]["out_feats"]) * config["model"]["graph_conv"]["out_feats"][0] + 1
        )
        self.cnn = CNN(
            in_channels=1,
            list_out_channels=config["model"]["convolutional"]["out_channels"],
            list_kernel_size=config["model"]["convolutional"]["kernel_size"],
            list_cnn_dropout=config["model"]["convolutional"]["cnn_dropout"],
            dim=1,
        )

        mlp_in_features = sum(config["model"]["graph_conv"]["out_feats"]) * config["model"]["sort_pooling"]["k"]
        for kernel_stride in config["model"]["convolutional"]["kernel_size"]:
            mlp_in_features -= kernel_stride - 1
        mlp_in_features *= config["model"]["convolutional"]["out_channels"][-1]
        self.mlp = MLP(
            in_features=mlp_in_features,
            list_out_features=config["model"]["linear"]["out_features"],
            list_linear_dropout=config["model"]["linear"]["dropout"],
        )

    def forward(
        self,
        graph: dgl.DGLHeteroGraph,
        node_labels: torch.Tensor,
        node_id: torch.Tensor,
        edge_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        forward pass of the SEAL model
        Apply:
         1. Graph layer
         2. SortPooling
         3. CNN
         4. MLP
        """
        device = graph.device
        node_feats = torch.cat(
            [
                self.node_attributes[node_id].to(device),
                self.node_embedding(node_id),
                self.node_embedding(self.node_attributes.shape[0] + node_labels),
            ],
            1,
        )

        # GNN
        node_feats = self.gnn(
            graph=graph,
            node_feats=node_feats,
            edge_weight=self.edge_weights[edge_id].to(device),
        )

        # ReLU
        node_feats = torch.relu(node_feats)

        # SortPooling
        node_feats = self.pooling(graph, node_feats)
        node_feats = node_feats.unsqueeze(1)

        # CNN
        node_feats = self.cnn(node_feats)
        node_feats = node_feats.view(node_feats.shape[0], -1)

        # MLP
        node_feats = self.mlp(node_feats)
        return node_feats
