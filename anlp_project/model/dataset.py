import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset

import random


class MELDText(Dataset):
    def __init__(self, split, model_name: str):
        dataset_path = os.path.join(
            os.path.dirname(__file__), f"../../dataset/{split}_sent_emo.csv"
        )
        self.dataset = pd.read_csv(dataset_path)
        self.dataset = self.dataset[["Utterance", "Emotion"]]
        self.author = "FacebookAI" if "bigbird" not in model_name else "google"
        self.info = (
            self.author + "/" + "-".join(model_name.split("/")[-1].split("-")[:-3])
            if len(model_name.split("/")[-1].split("-")) > 4
            else model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.info,
        )
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
        )
        text["input_ids"] = text["input_ids"].squeeze()
        text["attention_mask"] = text["attention_mask"].squeeze()
        label = torch.Tensor([self.order.index(item["Emotion"])]).squeeze()
        return text, label


class GoEmotionsMultiLabel(Dataset):
    def __init__(self, split, model_name):
        self.split = split
        self.dataset = load_dataset("go_emotions", "simplified", split=self.split)
        self.author = "FacebookAI" if "bigbird" not in model_name else "google"
        self.info = (
            self.author + "/" + "-".join(model_name.split("/")[-1].split("-")[:-3])
            if len(model_name.split("/")[-1].split("-")) > 4
            else model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.info)
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
        text = self.tokenizer(item["text"], return_tensors="pt", truncation=True)
        # [x,1,4096] to [x,4096]
        text = {k: v.squeeze() for k, v in text.items()}
        label = torch.Tensor([item["labels"]]).squeeze()
        return text, label

    def _reduce_labels(self, item):
        labels = list(
            set(
                [
                    self.new_order.index(self.annotaions[self.old_order[label]])
                    for label in item["labels"]
                ]
            )
        )

        item["labels"] = self._one_hot_encode(labels)

        return {k: v for k, v in item.items() if k != "_id"}

    def _one_hot_encode(self, item):
        return [1.0 if i in item else 0.0 for i in range(len(self.new_order))]


class GoEmotionsMultiClass(Dataset):
    def __init__(self, split, model_name):
        self.split = split
        self.dataset = load_dataset("go_emotions", "simplified", split=self.split)
        self.author = "FacebookAI" if "bigbird" not in model_name else "google"
        self.info = (
            self.author + "/" + "-".join(model_name.split("/")[-1].split("-")[:-3])
            if len(model_name.split("/")[-1].split("-")) > 4
            else model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.info)
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
        text = self.tokenizer(item["text"], return_tensors="pt", truncation=True)
        # [x,1,4096] to [x,4096]
        text = {k: v.squeeze() for k, v in text.items()}
        label = torch.Tensor([item["labels"]]).squeeze()
        return text, label

    def _reduce_labels(self, item):
        labels = list(
            set(
                [
                    self.new_order.index(self.annotaions[self.old_order[label]])
                    for label in item["labels"]
                ]
            )
        )

        if len(labels) > 1 and 6 in labels:
            labels.remove(6)

        if len(labels) > 1 and 5 in labels:
            labels.remove(5)

        item["labels"] = random.choice(labels)

        return {k: v for k, v in item.items() if k != "_id"}
