"""
Deep Learning Models that can be re-used in different use-cases
"""
from typing import List, Optional, Tuple, Union

import dgl
import torch
from dgl.nn.pytorch.conv import GraphConv
from torch import nn


class MLP(nn.Module):
    """
    Implementation of the Multilayer Perceptron model
    """

    def __init__(self, n_targets: int, n_features: int, hidden_nodes: Optional[List[int]] = None) -> None:
        super().__init__()

        if hidden_nodes is not None:
            layers = [
                nn.Linear(in_features=n_features, out_features=hidden_nodes[0]),
                nn.ReLU(),
            ]
            for enum, hidden in enumerate(hidden_nodes[1:]):
                layers.extend(
                    [
                        nn.Linear(in_features=hidden_nodes[enum], out_features=hidden),
                        nn.ReLU(),
                    ]
                )
            layers.append(nn.Linear(in_features=hidden_nodes[-1], out_features=n_targets))
        else:
            layers = [
                nn.Linear(in_features=n_features, out_features=n_targets),
            ]

        self.layers = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward of the MLP model
        :param data: torch.Tensor
        :return: torch.Tensor
        """
        return self.layers(data)


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
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                ]
            )
        cnn_layers = cnn_layers[:-2]  # remove last ReLU and Dropout
        self.cnn_layers = nn.Sequential(*cnn_layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward of the Convolutional Neural Network model
        :return: torch.Tensor
        """
        return self.cnn_layers(data)


class GraphEncoder(nn.Module):
    """
    Graph Encoder Module
    """

    def __init__(self, in_features: int, list_out_features: List[int], list_gnn_dropout: List[float]) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        num_layers = len(list_out_features)
        for enum, (out_feats, dropout) in enumerate(zip(list_out_features, list_gnn_dropout)):
            self.layers.extend(
                [
                    GraphConv(
                        in_feats=in_features if enum == 0 else list_out_features[enum - 1],
                        out_feats=out_feats,
                        norm="both",
                        weight=True,
                        bias=True,
                        activation=nn.ReLU() if enum < num_layers - 1 else None,
                        allow_zero_in_degree=False,
                    ),
                    nn.Dropout(p=dropout),
                ]
            )
        self.layers = self.layers[:-1]  # remove last Dropout

    def forward(self, feats: torch.Tensor, blocks: dgl.DGLHeteroGraph) -> torch.Tensor:
        """

        :param feats: torch.Tensor, node features
        :param blocks: dgl.DGLHeteroGraph, list of blocks (one per layer)
        :return: torch.Tensor, node embedding vectors
        """
        embeds = feats
        for layer, block in zip(self.layers, blocks):
            if isinstance(layer, GraphConv):
                embeds = layer(feats, block)
            else:
                embeds = layer(embeds)
        return embeds
