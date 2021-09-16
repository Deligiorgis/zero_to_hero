"""
Metrics used to evaluate the performances of the models
"""
from typing import Tuple, Union

import torch
from ogb.linkproppred import Evaluator
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
    with torch.no_grad():
        total_loss, accuracy = torch.zeros(1, device=device), torch.zeros(1, device=device)
        n_samples = 0
        for output in outputs:
            assert isinstance(output, dict)
            total_loss += output["losses"].sum()
            accuracy += (output["predictions"] == output["targets"].squeeze()).sum()
            n_samples += output["losses"].shape[0]
        return (total_loss / n_samples).cpu(), (accuracy / n_samples).cpu()


def compute_graph_metrics(
    outputs: EPOCH_OUTPUT,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """

    :param outputs: EPOCH_OUTPUT
    :param device: device on which the computations are executed
    :return: hits metric for collab
    """
    with torch.no_grad():
        total_loss, accuracy = torch.zeros(1, device=device), torch.zeros(1, device=device)
        assert isinstance(outputs[0], dict)
        pos_predictions, neg_predictions = outputs[0]["pos_predictions"], outputs[0]["neg_predictions"]
        n_samples = 0
        for enum, output in enumerate(outputs):
            assert isinstance(output, dict)
            total_loss += output["losses"].sum()
            accuracy += (output["predictions"] == output["targets"].squeeze()).sum()
            n_samples += output["losses"].shape[0]

            if enum > 0:
                pos_predictions = torch.cat([pos_predictions, output["pos_predictions"]])
                neg_predictions = torch.cat([neg_predictions, output["neg_predictions"]])

        evaluator = Evaluator(name="ogbl-collab")
        hits = evaluator.eval(
            {
                "y_pred_pos": pos_predictions.to(device),
                "y_pred_neg": neg_predictions.to(device),
            }
        )
        return total_loss / n_samples, accuracy / n_samples, hits
