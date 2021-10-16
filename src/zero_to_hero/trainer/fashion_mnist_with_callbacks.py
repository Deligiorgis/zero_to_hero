"""
Callbacks used in Trainer for fashionMNIST
"""
from typing import Dict

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def get_fashion_mnist_trainer_with_callbacks(config: Dict) -> Trainer:
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

    trainer = Trainer(
        gpus=[0] if torch.cuda.is_available() else None,
        max_epochs=config["hyper_parameters"]["epochs"],
        logger=logger,
        callbacks=[
            early_stop_callback,
            checkpoint_callback,
        ],
    )
    return trainer
