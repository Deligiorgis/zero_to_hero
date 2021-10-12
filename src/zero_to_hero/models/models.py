"""
Deep Learning Models that can be re-used in different use-cases
"""
from typing import Any, Dict, List, Tuple, Union

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

    def __init__(
        self,
        in_channels: int,
        list_out_channels: List[int],
        list_kernel_size: List[Union[int, Tuple[int, int]]],
        list_cnn_dropout: List[float],
    ) -> None:
        super().__init__()

        cnn_layers = []

        for enum, (out_channels, kernel_size, dropout) in enumerate(
            zip(list_out_channels, list_kernel_size, list_cnn_dropout)
        ):
            cnn_layers.extend(
                [
                    nn.Conv2d(
                        in_channels=in_channels if enum == 0 else list_out_channels[enum - 1],
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                    ),
                    nn.BatchNorm2d(num_features=out_channels),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                ]
            )
        cnn_layers = cnn_layers[:-3]  # remove last BatchNorm1d, ReLU and Dropout
        self.cnn_layers = nn.Sequential(*cnn_layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward of the Convolutional Neural Network model
        :return: torch.Tensor
        """
        return self.cnn_layers(data)


class DGCNN(nn.Module):
    """
    An end-to-end deep learning architecture for graph classification.
    paper link: https://muhanzhang.github.io/papers/AAAI_2018_DGCNN.pdf
    Attributes:
        num_layers(int): num of gcn layers
        hidden_units(int): num of hidden units
        k(int, optional): The number of nodes to hold for each graph in SortPooling.
        gcn_type(str): type of gcn layer, 'gcn' for GraphConv and 'sage' for SAGEConv
        node_attributes(Tensor, optional): node attribute
        edge_weights(Tensor, optional): edge weight
        node_embedding(Tensor, optional): pre-trained node embedding
        use_embedding(bool, optional): whether to use node embedding. Note that if 'use_embedding' is set True
                             and 'node_embedding' is None, will automatically randomly initialize node embedding.
        num_nodes(int, optional): num of nodes
        dropout(float, optional): dropout rate
        max_z(int, optional): default max vocab size of node labeling, default 1000.
    """

    def __init__(
        self,
        ndata: torch.Tensor,
        edata: torch.Tensor,
        config: Dict[str, Any],
    ):
        super().__init__()

        self.node_attributes_lookup = nn.Embedding.from_pretrained(ndata)
        self.node_attributes_lookup.weight.requires_grad = False

        self.edge_weights_lookup = nn.Embedding.from_pretrained(edata)
        self.edge_weights_lookup.weight.requires_grad = False

        self.node_embedding = nn.Embedding(ndata.shape[0] + config["max_z"], config["hidden_units"])

        initial_dim = self.node_attributes_lookup.embedding_dim + self.node_embedding.embedding_dim * 2

        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GraphConv(initial_dim, config["hidden_units"]))
        for _ in range(config["num_layers"] - 1):
            self.gcn_layers.append(GraphConv(config["hidden_units"], config["hidden_units"]))
        self.gcn_layers.append(GraphConv(config["hidden_units"], 1))

        self.pooling = SortPooling(k=config["k"])

        conv1d_channels = [16, 32]
        total_latent_dim = config["hidden_units"] * config["num_layers"] + 1
        conv1d_kws = [total_latent_dim, 5]
        dense_dim = int((config["k"] - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]

        self.cnn = nn.Sequential(
            nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0]),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(dense_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=config["dropout"]),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        graph: dgl.DGLHeteroGraph,
        feats: torch.Tensor,
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
        node_feats = torch.cat(
            [
                self.node_attributes_lookup(node_id),
                self.node_embedding(feats),
                self.node_embedding(self.node_attributes_lookup.num_embeddings + node_id),
            ],
            1,
        )

        list_node_feats = [node_feats]
        for layer in self.gcn_layers:
            list_node_feats += [
                torch.tanh(
                    layer(
                        graph=graph,
                        feat=list_node_feats[-1],
                        edge_weight=self.edge_weights_lookup(edge_id),
                    )
                )
            ]

        node_feats = torch.cat(list_node_feats[1:], dim=-1)
        del list_node_feats  # for memory efficiency

        # SortPooling
        node_feats = self.pooling(graph, node_feats)
        node_feats = node_feats.unsqueeze(1)

        # CNN
        node_feats = self.cnn(node_feats)
        node_feats = node_feats.view(node_feats.shape[0], -1)

        # MLP
        node_feats = self.mlp(node_feats)
        return node_feats
