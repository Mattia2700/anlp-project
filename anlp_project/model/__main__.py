import datasets
import wandb
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from anlp_project.model.dataset import GoEmotions
from anlp_project.model.model import LyricsClassifier


def main():
    wandb.finish()
    model_name, lr, num_labels, additional_layers = (
        "FacebookAI/roberta-base",
        1e-3,
        28,
        False,
    )
    model = LyricsClassifier(model_name, lr, num_labels, additional_layers)
    batch_size = 16
    train_data = GoEmotions(datasets.Split.TRAIN, transform=False)
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data = GoEmotions(datasets.Split.VALIDATION, transform=False)
    val_data = DataLoader(val_data, batch_size=batch_size)
    test_data = GoEmotions(datasets.Split.TEST, transform=False)
    test_data = DataLoader(test_data, batch_size=batch_size)

    logger = WandbLogger(
        project="anlp-project",
        name=f"{model_name.split('/')[1]}-{batch_size}-{lr}-{num_labels}",
    )
    logger.experiment.config["batch_size"] = batch_size

    trainer = Trainer(
        limit_train_batches=0.25,
        limit_val_batches=0.25,
        max_epochs=5,
        logger=logger,
    )

    trainer.fit(model, train_data, val_data)
    trainer.test(model, test_data)
