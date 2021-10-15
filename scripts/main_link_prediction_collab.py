"""
Main script to fit and predict the links (collaborations) between the authors
"""
import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch

from zero_to_hero.config_reader import read_config
from zero_to_hero.data.collab import CollabDataModule
from zero_to_hero.models.link_prediction_collab import LinkPredictorCollab
from zero_to_hero.trainer.collab_callbacks import get_collab_callbacks

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

    (
        logger,
        early_stop_callback,
        loss_checkpoint_callback,
        hits_checkpoint_callback,
    ) = get_collab_callbacks(config=config)
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
