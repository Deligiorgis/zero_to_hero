"""
Main script to fit and predict the label (classes) of the blobs
"""
import warnings
from pathlib import Path

import pytorch_lightning as pl

from zero_to_hero.config_reader import read_config
from zero_to_hero.data.blobs import BlobsDataModule
from zero_to_hero.models.blobs_classifier import BlobsClassifierModel
from zero_to_hero.trainer.blobs_with_callbacks import get_blobs_trainer_with_callbacks

warnings.filterwarnings("ignore")


def main() -> None:
    """
    Main function that runs the fit (training) of the blobs' classifier and its predictions
    :return:
    """
    pl.seed_everything(
        seed=1,
        workers=True,
    )

    config = read_config(path=Path("configs/blobs.yml"))

    datamodule = BlobsDataModule(config=config)
    model = BlobsClassifierModel(config=config)

    trainer = get_blobs_trainer_with_callbacks(config=config)

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

    predictions_probabilities = trainer.predict(
        model=model,
        datamodule=datamodule,
        return_predictions=True,
        ckpt_path="best",
    )
    print(predictions_probabilities)


if __name__ == "__main__":
    main()
