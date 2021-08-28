"""
Main script to fit and predict the label (classes) of the blobs
"""
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from zero_to_hero.config_reader import read_config
from zero_to_hero.data.blobs.datamodule import BlobsDataModule
from zero_to_hero.models.blobs.classifier import BlobsClassifierModel


def main() -> None:
    """
    Main function that runs the fit (training) of the blobs' classifier and its predictions
    :return:
    """
    config = read_config(path=Path("configs/blobs.yml"))

    datamodule = BlobsDataModule(config=config["data"])
    model = BlobsClassifierModel(
        n_features=config["data"]["n_features"],
        n_centers=len(config["data"]["centers"]),
        hidden_nodes=config["model"]["hidden_nodes"],
        learning_rate=config["model"]["learning_rate"],
    )

    logger = TensorBoardLogger(
        save_dir="tb_logs",
        name="my_model",
        default_hp_metric=False,
        prefix="blobs",
        sub_dir="blobs",
    )
    early_stop_callback = EarlyStopping(
        monitor="validation_loss",
        patience=config["hyper_parameters"]["patience"],
        verbose=False,
        mode="min",
        check_on_train_epoch_end=True,
    )
    trainer = pl.Trainer(
        gpus=[0],
        max_epochs=config["hyper_parameters"]["epochs"],
        logger=logger,
        callbacks=[
            early_stop_callback,
        ],
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    main()
