"""
Deep Learning Models that can be re-used in different use-cases
"""
from typing import List, Optional, Tuple, Union

import torch
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
    ) -> None:
        super().__init__()

        cnn_layers = []

        for enum, (out_channels, kernel_size) in enumerate(zip(list_out_channels, list_kernel_size)):
            cnn_layers.extend(
                [
                    nn.Conv2d(
                        in_channels=in_channels if enum == 0 else list_out_channels[enum - 1],
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                    ),
                    nn.ReLU(),
                ]
            )
        cnn_layers = cnn_layers[:-1]  # remove lust ReLU
        self.cnn_layers = nn.Sequential(*cnn_layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward of the Convolutional Neural Network model
        :return: torch.Tensor
        """
        return self.cnn_layers(data)
