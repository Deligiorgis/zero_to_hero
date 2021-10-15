"""
Callbacks used in Trainer for Blobs
"""
from abc import ABC
from typing import Dict, List

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def get_blobs_callbacks(config: Dict) -> List[ABC]:
    """

    Parameters
    ----------
    config

    Returns
    -------

    """
    logger = TensorBoardLogger(
        save_dir="tensorboard_logs",
        name="blobs",
        prefix=f"blobs--centers-{len(config['data']['centers'])}--dims-{config['data']['in_features']}",
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
        filename="blobs-classification-{epoch:02d}-{validation_loss:.4f}",
    )
    callbacks = [
        logger,
        early_stop_callback,
        checkpoint_callback,
    ]
    return callbacks
