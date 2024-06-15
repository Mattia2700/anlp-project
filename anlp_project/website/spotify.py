import datetime
import os
import re

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ["SPOTIPY_CLIENT_ID"] = "f516c752459a4b94acba6a768cad9c43"
os.environ["SPOTIPY_CLIENT_SECRET"] = "641839e196f24feda3341dd5ac972749"
os.environ["SPOTIPY_REDIRECT_URI"] = "http://localhost:5000/callback"
os.environ["GENIUS_CLIENT_ID"] = (
    "MzhBkTi6j7Ujvc6fk6h2AfJQZ_6jAShThC76DpUurH55n4lpiUK3eG8YaQX94wHp"
)

import spotipy
from spotipy.oauth2 import SpotifyOAuth

scope = "user-read-recently-played"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

last_week_timestamp = int(
    (datetime.datetime.now() - datetime.timedelta(weeks=1)).timestamp() * 1000
)

results = sp.current_user_recently_played(after=last_week_timestamp)

actual_songs = set()

for idx, item in enumerate(results["items"]):
    track = item["track"]
    actual_songs.add((track["artists"][0]["name"], track["name"]))

actual_songs = list(actual_songs)

for idx, (artist, song) in enumerate(actual_songs.copy()):
    if "- Remastered" in song:
        song = song.replace(" - Remastered", "")
    elif "- Radio Edit" in song:
        song = song.replace(" - Radio Edit", "")
    elif "- Remix" in song:
        song = song.replace(" - Remix", "")

    actual_songs[idx] = (artist, song)

print(actual_songs)

from lyricsgenius import Genius

genius = Genius(os.environ["GENIUS_CLIENT_ID"], remove_section_headers=True)

song = genius.search_song("Sunroof", "Nicky Youre", get_full_info=False)
# print(song.lyrics)
lyrics = [line for line in song.lyrics.split("\n")][1:]
# remove %dEmbed from last line
lyrics[-1] = re.sub(r"\d*K*Embed", "", lyrics[-1])

# parts = song.lyrics.split("\n\n")
# for idx, part in enumerate(parts.copy()):
#     if idx == 0:
#         lines = parts[idx].split("\n")
#         parts[idx] = "\n".join(lines[1:])
#     elif idx == len(parts) - 1:
#         # remove the last line that is the artist
#         lines = parts[idx].split("\n")
#         lines[-1] = re.sub(r"\d*K*Embed", "", lines[-1])
#         parts[idx] = "\n".join(lines)

#
# # model = LyricsClassifier()
# # model.load_state_dict(torch.load("/test/model.pth", map_location=torch.device("cpu")))
output_dir = "FacebookAI/roberta-large"
# # tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)

print(model)
exit()

# fmt: off
annotaions = {"anger": "anger", "annoyance": "anger", "disapproval": "anger", "disgust": "disgust", "fear": "fear", "nervousness": "fear", "joy": "joy", "amusement": "joy", "approval": "joy", "excitement": "joy", "gratitude": "joy", "love": "joy", "optimism": "joy", "relief": "joy", "pride": "joy", "admiration": "joy", "desire": "joy", "caring": "joy", "sadness": "sadness", "disappointment": "sadness", "embarrassment": "sadness", "grief": "sadness", "remorse": "sadness", "surprise": "surprise", "realization": "surprise", "confusion": "surprise", "curiosity": "surprise", "neutral": "neutral"}
old_order = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]
new_order = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
# fmt: on

#
# # print(model)
#
moods = []
#
# # text = tokenizer(
# #     "i am glad you came!",
# #     return_tensors="pt",
# #     padding="max_length",
# #     truncation=True,
# # )
# # output = model(**text)
# # print(output.logits)
# # mood = torch.argmax(output.logits).item()
# # print(mood)
#
for part in lyrics:
    # print(line)
    text = tokenizer(part, return_tensors="pt")
    output = model(**text)
    # print(output)
    mood = torch.argmax(output.logits).item()
    # mood = new_order.index(annotaions[old_order[mood]])
    moods.append(mood)
    print(f"{part}\n---\n{mood}\n---\n")
#
print(moods)
print(max(set(moods), key=moods.count))
moods = [mood for mood in moods if mood != 4]
# # # print the number that appears the most
print(moods)
print(max(set(moods), key=moods.count))
#
#
# # # print available genres
# print(sp.recommendation_genre_seeds())
#
# # 5 genres
# res = sp.recommendations(seed_genres=["pop", "rock"], limit=5)
#
# for idx, item in enumerate(res['tracks']):
#     print(item['name'], '-', item['artists'][0]['name'])
