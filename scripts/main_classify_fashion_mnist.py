"""
Main script to train and classify the clothes from FashionMNIST
"""
import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch
from pl_bolts.datamodules import FashionMNISTDataModule

from zero_to_hero.config_reader import read_config
from zero_to_hero.models.fashion_mnist_classifier import FashionMNISTClassifier
from zero_to_hero.trainer.fashion_mnist_callbacks import get_fashion_mnist_callbacks

warnings.filterwarnings("ignore")


def main() -> None:
    """
    Implementation of the main function that trains and test a classifier on the FashionMNIST data
    :return:
    """
    pl.seed_everything(
        seed=1,
        workers=True,
    )

    config = read_config(path=Path("configs/fashion_mnist.yml"))

    datamodule = FashionMNISTDataModule(
        data_dir="data",
        val_split=config["data"]["val_split"],
        num_workers=config["hyper_parameters"]["num_workers"],
        batch_size=config["hyper_parameters"]["batch_size"],
        pin_memory=config["hyper_parameters"]["pin_memory"],
        normalize=True,
        shuffle=True,
        drop_last=False,
        seed=1,
    )
    model = FashionMNISTClassifier(configs=config)

    (
        logger,
        early_stop_callback,
        checkpoint_callback,
    ) = get_fashion_mnist_callbacks(config=config)
    trainer = pl.Trainer(
        gpus=[0] if torch.cuda.is_available() else None,
        max_epochs=config["hyper_parameters"]["epochs"],
        logger=logger,
        callbacks=[
            early_stop_callback,
            checkpoint_callback,
        ],
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )
    print("Best checkpoint path:", trainer.checkpoint_callback.best_model_path)

    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path="best",
        verbose=True,
    )


if __name__ == "__main__":
    main()
