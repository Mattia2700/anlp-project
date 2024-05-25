import os
import re
import datetime

os.environ["SPOTIPY_CLIENT_ID"] = "f516c752459a4b94acba6a768cad9c43"
os.environ["SPOTIPY_CLIENT_SECRET"] = "641839e196f24feda3341dd5ac972749"
os.environ["SPOTIPY_REDIRECT_URI"] = "http://localhost:5000/callback"
os.environ["GENIUS_CLIENT_ID"] = "MzhBkTi6j7Ujvc6fk6h2AfJQZ_6jAShThC76DpUurH55n4lpiUK3eG8YaQX94wHp"

import spotipy
from spotipy.oauth2 import SpotifyOAuth

scope = "user-read-recently-played"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

last_week_timestamp = int((datetime.datetime.now() - datetime.timedelta(weeks=1)).timestamp() * 1000)

results = sp.current_user_recently_played(after=last_week_timestamp)

actual_songs = set()

for idx, item in enumerate(results["items"]):
    track = item["track"]
    actual_songs.add((track["artists"][0]["name"], track["name"]))

actual_songs = list(actual_songs)

for idx, (artist, song) in enumerate(actual_songs.copy()):
    if '- Remastered' in song:
        song = song.replace(' - Remastered', '')
    elif '- Radio Edit' in song:
        song = song.replace(' - Radio Edit', '')
    elif '- Remix' in song:
        song = song.replace(' - Remix', '')

    actual_songs[idx] = (artist, song)

print(actual_songs)

# from lyricsgenius import Genius
#
# genius = Genius(os.environ["GENIUS_CLIENT_ID"])
# genius.remove_section_headers = True
#
# song = genius.search_song(title, artist, get_full_info=False)
# lyrics = [line for line in song.lyrics.split("\n") if line != ""][1:]
# # remove %dEmbed from last line
# lyrics[-1] = re.sub(r"\d*Embed", "", lyrics[-1])
#
# print('\n'.join(lyrics))


# # print available genres
# print(sp.recommendation_genre_seeds())
#
# # 5 genres
# res = sp.recommendations(seed_genres=["pop", "rock"], limit=5)
#
# for idx, item in enumerate(res['tracks']):
#     print(item['name'], '-', item['artists'][0]['name'])