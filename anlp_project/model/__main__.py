from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks import StochasticWeightAveraging, LearningRateFinder
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


def main():
    model_name, lr, num_labels, batch_size = (
        "FacebookAI/xlm-roberta-base",
        5e-6,
        7,
        32,
    )

    model = LyricsClassifier(model_name, lr, num_labels, batch_size)
    # checkpoint_callback = ModelCheckpoint(dirpath="model", save_top_k=2, monitor="val_loss")

    epochs = 20

    logger = WandbLogger(
        project="anlp-project",
        name=f"{model_name.split('/')[1]}-{batch_size}-{lr}-{num_labels}-{epochs}",
    )

    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
        # callbacks=[
        #     StochasticWeightAveraging(swa_lrs=1e-2),
        # ],
    )
    # tuner = Tuner(trainer)
    # tuner.lr_find(model)

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

if __name__ == "__main__":
    main()