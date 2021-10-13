"""
Main script to fit and predict the label (classes) of the blobs
"""
import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch.cuda
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from zero_to_hero.config_reader import read_config
from zero_to_hero.data.blobs import BlobsDataModule
from zero_to_hero.models.blobs_classifier import BlobsClassifierModel

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

    predictions_probabilities = trainer.predict(
        model=model,
        datamodule=datamodule,
        return_predictions=True,
        ckpt_path="best",
    )
    print(predictions_probabilities)


if __name__ == "__main__":
    main()
