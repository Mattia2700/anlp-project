import re

import torch
from lyricsgenius import Genius
from transformers import AutoTokenizer

from anlp_project.model.model import LyricsClassifier


class MoodHelper:
    model = LyricsClassifier.load_from_checkpoint(
        "/home/dhilab-mattia/Downloads/epoch=19-step=27140-7.ckpt"
    )
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    genius = Genius(
        "MzhBkTi6j7Ujvc6fk6h2AfJQZ_6jAShThC76DpUurH55n4lpiUK3eG8YaQX94wHp",
        remove_section_headers=True,
    )
    threshold = 0.5
    emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

    @staticmethod
    def get_mood(songs: list):
        moods = [0] * 7
        songs = [songs[0]]
        # print(songs)
        for song in songs:
            lyrics = MoodHelper.get_song_lyrics("Another Lover", "Another Love")
            for line in lyrics:
                encoded = MoodHelper.tokenizer(line, return_tensors="pt")
                logits = MoodHelper.model(encoded)
                print(line)
                print(logits)
                print("----")
                indexes = torch.nonzero(logits > MoodHelper.threshold)
                values = logits[logits > MoodHelper.threshold]

                if indexes.numel() == 0:
                    continue

                for idx, value in zip(indexes, values):
                    moods[idx] += value.item()

        print(moods)

        # get top 2 moods
        top_moods = sorted(
            [(mood, idx) for idx, mood in enumerate(moods)], reverse=True
        )[:2]

        # returns the second mood if the first mood is neutral and the second mood is non-zero
        if top_moods[0][1] == 6 and top_moods[0][0] > 0:
            return MoodHelper.emotions[(top_moods[1][1])]
        else:
            return MoodHelper.emotions[top_moods[0][1]]

    @staticmethod
    def get_song_lyrics(artist: str, song: str):
        song = MoodHelper.genius.search_song(song, artist, get_full_info=False)
        lyrics = [line for line in song.lyrics.split("\n")][1:]
        lyrics[-1] = re.sub(r"\d*K*Embed", "", lyrics[-1])
        lyrics[-1] = re.sub(r"You might also like", "", lyrics[-1])
        lyrics = [line for line in lyrics if line != ""]
        # group the lines by 2 lines
        lyrics = [
            lyrics[i] + " " + lyrics[i + 1] + " " + lyrics[i + 2] + " " + lyrics[i + 3]
            for i in range(len(lyrics) - 3)
        ]
        return lyrics
