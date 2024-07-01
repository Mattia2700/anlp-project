import click
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import LearningRateFinder
from pytorch_lightning.loggers import WandbLogger

from anlp_project.model.model import LyricsClassifier

import torch


torch.set_float32_matmul_precision("high")


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


@click.option("--train", is_flag=True, default=False, help="Whether to train the model")
@click.option("--epochs", default=12, help="Number of epochs to train the model")
@click.option(
    "--model-name",
    default="Franzin/xlm-roberta-base-goemotions-ekman-multilabel",
    help="Model name",
)
@click.option("--lr", default=5e-6, help="Learning rate")
@click.option("--num-labels", default=7, help="Number of labels")
@click.option("--batch-size", default=32, help="Batch size")
@click.option("--dataset", default="goemotions", help="Dataset to use")
@click.command()
def main(train, epochs, model_name, lr, num_labels, batch_size, dataset):
    model = LyricsClassifier(model_name, lr, num_labels, batch_size, dataset)

    logger = WandbLogger(
        project="anlp-project",
        name=f"{model_name.split('/')[1]}-{batch_size}-{lr}-{num_labels}-{epochs}",
    )

    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
    )

    # logger.experiment.config["batch_size"] = model.batch_size

    if train:
        trainer.fit(model, model.train_dataloader(), model.val_dataloader())

    trainer.test(model, model.test_dataloader())


if __name__ == "__main__":
    main()
