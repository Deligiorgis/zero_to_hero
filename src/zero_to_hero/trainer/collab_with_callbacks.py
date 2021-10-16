"""
Callbacks used in Trainer for Collab
"""
from typing import Dict

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def get_collab_trainer_with_callbacks(config: Dict) -> Trainer:
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

    trainer = Trainer(
        gpus=[0] if torch.cuda.is_available() else None,
        max_epochs=config["hyper_parameters"]["epochs"],
        logger=logger,
        callbacks=[
            early_stop_callback,
            loss_checkpoint_callback,
            hits_checkpoint_callback,
        ],
        limit_train_batches=100,
        limit_val_batches=100,
        limit_test_batches=100,
    )
    return trainer
