"""
Callbacks used in Trainer for fashionMNIST
"""
from abc import ABC
from typing import Dict, List

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def get_fashion_mnist_callbacks(config: Dict) -> List[ABC]:
    """

    Parameters
    ----------
    config

    Returns
    -------

    """
    logger = TensorBoardLogger(
        save_dir="tensorboard_logs",
        name="fashionMNIST",
        prefix="fashionMNIST",
        default_hp_metric=False,
        log_graph=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="validation_loss",
        patience=config["hyper_parameters"]["patience"],
        verbose=False,
        mode="min",
        check_on_train_epoch_end=True,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="validation_loss",
        mode="min",
        verbose=False,
        save_last=True,
        save_top_k=1,
        filename="fashionMNIST-classification-{epoch:02d}-{validation_loss:.4f}",
    )
    callbacks = [
        logger,
        early_stop_callback,
        checkpoint_callback,
    ]
    return callbacks
