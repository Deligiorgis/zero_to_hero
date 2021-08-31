"""
Functions that facilitate the logging of the model in different stages and steps
"""
from typing import Dict, List

import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT


def get_data_from_outputs(
    keys: List[str],
    outputs: EPOCH_OUTPUT,
) -> Dict[str, torch.Tensor]:
    """
    Get data and targets from outputs that have been gathered from the batches
    :param keys: Which data to fetch from outputs
    :param outputs: EPOCH_OUTPUT, dictionary that contains necessary information
    :return: data: torch.Tensor, targets: torch.Tensor, predictions: torch.Tensor
    """
    assert isinstance(outputs[0], dict)
    data = {key: outputs[0][key] for key in keys}
    for output in outputs[1:]:
        assert isinstance(output, dict)
        for key in keys:
            data[key] = torch.cat([data[key], output[key]], dim=0)
    return data
