"""
Callbacks used in Trainer for Collab
"""
from abc import ABC
from typing import Dict, List

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def get_collab_callbacks(config: Dict) -> List[ABC]:
    """

    Parameters
    ----------
    config

    Returns
    -------

    """
    logger = TensorBoardLogger(
        save_dir="tensorboard_logs",
        name="collab",
        prefix="collab",
        default_hp_metric=False,
        log_graph=False,
    )
    early_stop_callback = EarlyStopping(
        monitor="validation_hits",
        patience=config["hyper_parameters"]["patience"],
        verbose=True,
        mode="max",
        check_on_train_epoch_end=True,
    )
    loss_checkpoint_callback = ModelCheckpoint(
        monitor="validation_loss",
        mode="min",
        verbose=True,
        save_last=True,
        save_top_k=1,
        filename="collab-link-prediction-{epoch:02d}-{validation_loss:.4f}",
    )
    hits_checkpoint_callback = ModelCheckpoint(
        monitor="validation_hits",
        mode="max",
        verbose=True,
        save_last=True,
        save_top_k=1,
        filename="collab-link-prediction-{epoch:02d}-{validation_loss:.4f}",
    )

    callbacks = [
        logger,
        early_stop_callback,
        loss_checkpoint_callback,
        hits_checkpoint_callback,
    ]
    return callbacks
