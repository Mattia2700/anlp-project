import os
from typing import Literal

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


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
