import os
from typing import Literal

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset

class MELDText(Dataset):
    def __init__(self, split: Literal["train", "dev", "test"], model: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset_path = os.path.join(
            os.path.dirname(__file__), f"../../dataset/{split}_sent_emo.csv"
        )
        self.dataset = pd.read_csv(dataset_path)
        self.dataset = self.dataset[["Utterance", "Emotion"]]
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.order = [
            "anger",
            "disgust",
            "fear",
            "joy",
            "sadness",
            "surprise",
            "neutral",
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset.iloc[idx]
        text = self.tokenizer(
            item["Utterance"],
            return_tensors="pt",
        ).to(self.device)
        text["input_ids"] = text["input_ids"].squeeze()
        text["attention_mask"] = text["attention_mask"].squeeze()
        label = self._one_hot_encode(self.order.index(item["Emotion"])).to(self.device)
        return text, label

    def _one_hot_encode(self, item):
        return torch.Tensor([1 if i == item else 0 for i in range(7)])


class GoEmotions(Dataset):
    def __init__(self, split, model_name):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = load_dataset("go_emotions", "simplified", split=split)
        self.dataset = self.dataset.map(self._drop_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if True:
            # fmt: off
            self.annotaions = {"anger": "anger", "annoyance": "anger", "disapproval": "anger", "disgust": "disgust", "fear": "fear", "nervousness": "fear", "joy": "joy", "amusement": "joy", "approval": "joy", "excitement": "joy", "gratitude": "joy", "love": "joy", "optimism": "joy", "relief": "joy", "pride": "joy", "admiration": "joy", "desire": "joy", "caring": "joy", "sadness": "sadness", "disappointment": "sadness", "embarrassment": "sadness", "grief": "sadness", "remorse": "sadness", "surprise": "surprise", "realization": "surprise", "confusion": "surprise", "curiosity": "surprise", "neutral": "neutral"}
            self.old_order = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]
            self.new_order = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
            # fmt: on
            self.dataset = self.dataset.map(self._reduce_labels)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = self.tokenizer(item["text"], return_tensors="pt")
        # [x,1,4096] to [x,4096]
        text = {k: v.squeeze() for k, v in text.items()}
        label = self._one_hot_encode(item["labels"])
        return text, label

    def _one_hot_encode(self, item):
        return torch.Tensor([1 if i in item else 0 for i in range(7)])

    def _reduce_labels(self, item):
        item["labels"] = list(
            set(
                [
                    self.new_order.index(self.annotaions[self.old_order[label]])
                    for label in item["labels"]
                ]
            )
        )
        # print(item["labels"])
        return {k: v for k, v in item.items() if k != "id"}

    def _drop_id(self, item):
        return
