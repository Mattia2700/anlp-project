import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import StochasticWeightAveraging, LearningRateFinder
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from anlp_project.model.dataset import MELDText
from anlp_project.model.model import LyricsClassifier
from pytorch_lightning.tuner import Tuner

from transformers import AutoTokenizer

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
    model_name, lr, num_labels, batch_size = ("FacebookAI/xlm-roberta-large", 1e-3, 7, 256)
    model = LyricsClassifier(model_name, lr, num_labels, batch_size)
    # checkpoint_callback = ModelCheckpoint(dirpath="model", save_top_k=2, monitor="val_loss")

    epochs = 20

    logger = WandbLogger(
        project="anlp-project",
        name=f"{model_name.split('/')[1]}-{batch_size}-{lr}-{num_labels}",
    )

    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
    )

    logger.experiment.config["batch_size"] = model.batch_size

    trainer.fit(model, model.train_dataloader(), model.val_dataloader())
    trainer.test(model, model.test_dataloader())
    # model = LyricsClassifier.load_from_checkpoint("/teamspace/studios/this_studio/anlp-project/anlp-project/frtm0jqj/checkpoints/epoch=19-step=3047.ckpt")
    # model.eval()
    # tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")

    # text = input()

    # while text != "exit":
    #     print(text)
    #     tok = tokenizer(text, return_tensors="pt")
    #     val = model(tok)
    #     print(val)
    #     text = input()