"""
Metrics used to evaluate the performances of the models
"""
from typing import Tuple, Union

import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT


def compute_metrics(
    outputs: EPOCH_OUTPUT,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute total metrics
    :param outputs: EPOCH_OUTPUT
    :param device: device on which the computations are executed
    :return: average loss: float, average accuracy: float
    """
    total_loss, accuracy = torch.zeros(1, device=device), torch.zeros(1, device=device)
    n_samples = 0
    for output in outputs:
        assert isinstance(output, dict)
        total_loss += output["losses"].sum()
        accuracy += (output["predictions"] == output["targets"].squeeze()).sum()
        n_samples += output["losses"].shape[0]
    return total_loss / n_samples, accuracy / n_samples
