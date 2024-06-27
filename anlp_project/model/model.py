import torch
from pytorch_lightning import LightningModule
from torchmetrics import F1Score, Accuracy
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from anlp_project.model.dataset import GoEmotionsWhy


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
            problem_type="multi_label_classification",
        )

        self.save_hyperparameters()

        self.val_f1 = F1Score(
            task="multilabel", num_labels=self.num_labels, average="macro"
        )
        self.test_f1 = F1Score(
            task="multilabel", num_labels=self.num_labels, average="macro"
        )

        self.val_acc = Accuracy(
            task="multilabel", num_labels=self.num_labels, average="macro"
        )
        self.test_acc = Accuracy(
            task="multilabel", num_labels=self.num_labels, average="macro"
        )

    def forward(self, x, labels=None):
        input_ids, attention_mask = x["input_ids"], x["attention_mask"]
        x = self.model(input_ids, attention_mask, labels=labels)
        return x

    def on_train_epoch_start(self):
        # freeze he model after 8 epochs
        if self.current_epoch == 8:
            for name, param in self.model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
        

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x, labels=y.float())
        self.log("train/loss", outputs.loss, on_step=True, on_epoch=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x, labels=y.float())
        self.log("val/loss", outputs.loss, on_epoch=True)
        self.log("val/f1-macro", self.val_f1(outputs.logits, y), on_epoch=True)
        self.log("val/acc-macro", self.val_acc(outputs.logits, y), on_epoch=True)
        return outputs.loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x, labels=y.float())
        self.log("test/loss", outputs.loss, on_epoch=True)
        self.log("test/f1-macro", self.test_f1(outputs.logits, y), on_epoch=True)
        self.log("test/acc-macro", self.test_acc(outputs.logits, y), on_epoch=True)
        return outputs.loss

    def train_dataloader(self):
        train_data = GoEmotionsWhy("train", self.model_name)
        return DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=8,
        )

    def val_dataloader(self):
        val_data = GoEmotionsWhy("validation", self.model_name)
        return DataLoader(
            val_data,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=8,
        )

    def test_dataloader(self):
        test_data = GoEmotionsWhy("test", self.model_name)
        return DataLoader(
            test_data,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=8,
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
        y = y.long().squeeze()
        return x, y
