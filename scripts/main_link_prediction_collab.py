"""
Main script to fit and predict the links (collaborations) between the authors
"""
import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from zero_to_hero.config_reader import read_config
from zero_to_hero.data.collab import CollabDataModule
from zero_to_hero.models.link_prediction_collab import LinkPredictorCollab

warnings.filterwarnings("ignore")


def main() -> None:
    """

    :return:
    """
    pl.seed_everything(
        seed=1,
        workers=True,
    )

    config = read_config(path=Path("configs/collab.yml"))

    datamodule = CollabDataModule(config=config)
    datamodule.prepare_data()
    datamodule.setup()

    model = LinkPredictorCollab(
        ndata=datamodule.ndata,
        edata=datamodule.edata,
        config=config,
    )

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
    trainer = pl.Trainer(
        gpus=[0] if torch.cuda.is_available() else None,
        max_epochs=config["hyper_parameters"]["epochs"],
        logger=logger,
        callbacks=[
            early_stop_callback,
            loss_checkpoint_callback,
            hits_checkpoint_callback,
        ],
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )
    print("Best checkpoint path:", trainer.checkpoint_callback.best_model_path)

    model = LinkPredictorCollab.load_from_checkpoint(
        checkpoint_path=trainer.checkpoint_callback.best_model_path,
        map_location="cpu",
        ndata=datamodule.ndata,
        edata=datamodule.test_edata,
        config=config,
    )
    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path="best",
        verbose=True,
    )


if __name__ == "__main__":
    main()
