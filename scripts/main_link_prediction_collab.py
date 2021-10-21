"""
Main script to fit and predict the links (collaborations) between the authors
"""
import warnings
from pathlib import Path

import pytorch_lightning as pl

from zero_to_hero.config_reader import read_config
from zero_to_hero.data.collab import CollabDataModule
from zero_to_hero.models.link_prediction_collab import LinkPredictorCollab
from zero_to_hero.trainer.collab_with_callbacks import get_collab_trainer_with_callbacks

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
        edata=datamodule.edata["fit"],
        config=config,
    )

    trainer = get_collab_trainer_with_callbacks(config=config)

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )
    print("Best checkpoint path:", trainer.checkpoint_callback.best_model_path)

    trainer.validate(
        model=model,
        datamodule=datamodule,
        ckpt_path="best",
        verbose=True,
    )

    model = LinkPredictorCollab.load_from_checkpoint(
        checkpoint_path=trainer.checkpoint_callback.best_model_path,
        map_location="cpu",
        ndata=datamodule.ndata,
        edata=datamodule.edata["test"],
        config=config,
    )
    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path="best",
        verbose=True,
    )

    model = LinkPredictorCollab.load_from_checkpoint(
        checkpoint_path=trainer.checkpoint_callback.best_model_path,
        map_location="cpu",
        ndata=datamodule.ndata,
        edata=datamodule.edata["predict"],
        config=config,
    )
    predictions = trainer.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path="best",
        return_predictions=True,
    )
    print(predictions)


if __name__ == "__main__":
    main()
