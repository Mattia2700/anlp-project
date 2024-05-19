import os

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


class GoEmotions(Dataset):
    def __init__(self, split, transform=True):
        self.dataset = load_dataset("go_emotions", "simplified", split="train")
        if transform:
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
        # TODO: tokenization
        return item["text"], item["labels"]

    def _reduce_labels(self, item):
        item["labels"] = list(
            set(
                [
                    self.new_order.index(self.annotaions[self.old_order[label]])
                    for label in item["labels"]
                ]
            )
        )
        return item


def main():
    x = GoEmotions("train")
    print(x[0])
