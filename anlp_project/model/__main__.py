import os

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import StochasticWeightAveraging, LearningRateFinder
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from anlp_project.model.dataset import MELDText
from anlp_project.model.model import LyricsClassifier


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


def collate_fn(batch):
    x, y = zip(*batch)
    # get the max length of the input_ids
    max_len = max([len(i["input_ids"]) for i in x])
    # pad the input_ids and attention_mask
    for i in range(len(x)):
        x[i]["input_ids"] = x[i]["input_ids"].tolist() + [0] * (
            max_len - len(x[i]["input_ids"])
        )
        x[i]["attention_mask"] = x[i]["attention_mask"].tolist() + [0] * (
            max_len - len(x[i]["attention_mask"])
        )
    x = {k: torch.tensor([i[k] for i in x]) for k in x[0].keys()}
    y = torch.stack(y)
    return x, y


def main():
    wandb.finish()
    model_name, lr, num_labels = ("FacebookAI/roberta-base", 1e-3, 7)
    model = LyricsClassifier(model_name, lr, num_labels)
    batch_size = 32
    train_data = MELDText("train", model_name)
    train_data = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=os.cpu_count(),
    )
    val_data = MELDText("dev", model_name)
    val_data = DataLoader(
        val_data,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=os.cpu_count(),
    )
    test_data = MELDText("test", model_name)
    test_data = DataLoader(
        test_data,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=os.cpu_count(),
    )

    logger = WandbLogger(
        project="anlp-project",
        name=f"{model_name.split('/')[1]}-{batch_size}-{lr}-{num_labels}",
    )
    logger.experiment.config["batch_size"] = batch_size

    epochs = 10

    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
        callbacks=[
            StochasticWeightAveraging(swa_lrs=1e-2),
            FineTuneLearningRateFinder(milestones=(range(epochs))),
        ],
    )

    trainer.fit(model, train_data, val_data)
    trainer.test(model, test_data)
