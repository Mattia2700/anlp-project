import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import StochasticWeightAveraging, LearningRateFinder
from pytorch_lightning.loggers import WandbLogger

from anlp_project.model.dataset import MELDText
from anlp_project.model.model import LyricsClassifier
from pytorch_lightning.tuner import Tuner


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


def main():
    model_name, lr, num_labels, batch_size = ("FacebookAI/roberta-base", 1e-3, 7, 64)
    model = LyricsClassifier(model_name, lr, num_labels, batch_size)

    epochs = 10

    logger = WandbLogger(
        project="anlp-project",
        name=f"{model_name.split('/')[1]}-{batch_size}-{lr}-{num_labels}",
    )
    logger.experiment.config["batch_size"] = batch_size

    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
        callbacks=[
            StochasticWeightAveraging(swa_lrs=1e-2),
            FineTuneLearningRateFinder(milestones=(range(epochs))),
        ],
    )

    tuner = Tuner(trainer)
    tuner.scale_batch_size(model, mode='binsearch')

    trainer.fit(model, model.train_dataloader(), model.val_dataloader())
    trainer.test(model, model.test_dataloader())
