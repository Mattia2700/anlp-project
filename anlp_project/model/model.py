import torch
import os
from pytorch_lightning import LightningModule
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from anlp_project.model.old_dataset import GoEmotions


class LyricsClassifier(LightningModule):
    def __init__(self, model_name, lr, num_labels, batch_size):
        super().__init__()
        self.model_name = model_name
        self.lr = lr
        self.num_labels = num_labels
        self.batch_size = batch_size

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
        ).to(self.device)
        self.save_hyperparameters()

    def forward(self, x):
        input_ids, attention_mask = x["input_ids"], x["attention_mask"]
        x = self.model(input_ids, attention_mask).logits
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
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

    def train_dataloader(self):
        train_data = GoEmotions("train", self.model_name)
        return DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        val_data = GoEmotions("validation", self.model_name)
        return DataLoader(
            val_data,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )
    
    def test_dataloader(self):
        test_data = GoEmotions("test", self.model_name)
        return DataLoader(
            test_data,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        x, y = zip(*batch)
        # get the max length of the input_ids
        max_len = max([len(i["input_ids"]) for i in x])
        # pad the input_ids and acollate_fn,ttention_mask
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