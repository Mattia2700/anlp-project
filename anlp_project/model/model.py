import torch
from lightning import LightningModule
from transformers import AutoModelForSequenceClassification


class LyricsClassifier(LightningModule):
    def __init__(
        self,
        model_name,
        lr,
        num_labels,
        additional_layers,
    ):
        super().__init__()
        self.model_name = model_name
        self.lr = lr
        self.num_labels = num_labels
        self.additional_layers = additional_layers
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="multi_label_classification",
        )
        self.save_hyperparameters()

    def forward(self, x):
        input_ids, attention_mask = x["input_ids"], x["attention_mask"]
        x = self.model(input_ids, attention_mask).logits
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(**x, labels=y)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(**x, labels=y)
        loss = outputs.loss
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(**x, labels=y)
        loss = outputs.loss
        self.log("test_loss", loss)
        return loss
