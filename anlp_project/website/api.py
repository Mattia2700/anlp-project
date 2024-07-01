import datetime
import os
import random
import uuid

import spotipy
import uvicorn
from fastapi import FastAPI
from spotipy.oauth2 import SpotifyOAuth
from starlette.requests import Request
from starlette.responses import RedirectResponse, FileResponse
from starlette.staticfiles import StaticFiles

from anlp_project.website.inference import MoodHelper

os.environ["SPOTIPY_CLIENT_ID"] = "f516c752459a4b94acba6a768cad9c43"
os.environ["SPOTIPY_CLIENT_SECRET"] = "641839e196f24feda3341dd5ac972749"
os.environ["SPOTIPY_REDIRECT_URI"] = "http://localhost:5000/callback"

app = FastAPI()
app.scope = "user-read-recently-played playlist-modify-private"
app.sp = SpotifyOAuth(scope=app.scope)
app.client = None
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/api/authorize")
def spotify_authorize(request: Request):
    return RedirectResponse(app.sp.get_authorize_url())


@app.get("/api/recent")
def get_recent(request: Request):
    access_token = None
    code = app.sp.parse_response_code(request.query_params["code"])
    if code:
        token_info = app.sp.get_access_token(code)
        access_token = token_info["access_token"]

    if access_token:
        print(access_token)
        app.client = spotipy.Spotify(auth=access_token)
        last_week_timestamp = int(
            (datetime.datetime.now() - datetime.timedelta(weeks=1)).timestamp() * 1000
        )

        results = app.client.current_user_recently_played(
            after=last_week_timestamp, limit=10
        )

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
            elif "(" in song and ")" in song:
                # remove brackets and text inside
                song = song[: song.index("(") - 1] + song[song.index(")") + 1 :].strip()

            actual_songs[idx] = (artist, song)

        return {"songs": actual_songs}


@app.post("/api/mood")
async def get_mood(request: Request):
    songs = await request.json()
    emotion = MoodHelper.get_mood(songs["songs"])
    return {"mood": emotion}


@app.post("/api/suggest")
async def suggest_songs(request: Request):
    body = await request.json()
    client: spotipy.Spotify = app.client
    genres = random.sample(client.recommendation_genre_seeds()["genres"], 5)
    min_valence, max_valence = 0.0, 1.0
    min_energy, max_energy = 0.0, 1.0
    min_danceability, max_danceability = 0.0, 1.0
    # ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

    match body["mood"]:
        case "anger":
            min_valence, max_valence = 0.0, 0.3
            min_energy, max_energy = 0.7, 1.0
            min_danceability, max_danceability = 0.3, 0.6
        case "disgust":
            min_valence, max_valence = 0.0, 0.3
            min_energy, max_energy = 0.3, 0.6
            min_danceability, max_danceability = 0.0, 0.3
        case "fear":
            min_valence, max_valence = 0.0, 0.3
            min_energy, max_energy = 0.6, 1.0
            min_danceability, max_danceability = 0.0, 0.4
        case "joy":
            min_valence, max_valence = 0.7, 1.0
            min_energy, max_energy = 0.7, 1.0
            min_danceability, max_danceability = 0.6, 1.0
        case "sadness":
            min_valence, max_valence = 0.0, 0.3
            min_energy, max_energy = 0.0, 0.4
            min_danceability, max_danceability = 0.0, 0.4
        case "surprise":
            min_valence, max_valence = 0.5, 1.0
            min_energy, max_energy = 0.7, 1.0
            min_danceability, max_danceability = 0.4, 0.7
        case "neutral":
            min_valence, max_valence = 0.4, 0.6
            min_energy, max_energy = 0.4, 0.6
            min_danceability, max_danceability = 0.4, 0.6
        case _:
            raise ValueError("Invalid mood")

    suggestions = client.recommendations(
        seed_genres=genres,
        min_valence=min_valence,
        max_valence=max_valence,
        min_energy=min_energy,
        max_energy=max_energy,
        min_danceability=min_danceability,
        max_danceability=max_danceability,
        country="IT",
        limit=10,
    )

    suggested_songs = [
        {
            "artist": ", ".join([artist["name"] for artist in track["artists"]]),
            "song": track["name"],
            "id": track["id"],
        }
        for track in suggestions["tracks"]
    ]

    return {"suggestions": suggested_songs}


@app.post("/api/create")
async def create_playlist(request: Request):
    body = await request.json()
    client: spotipy.Spotify = app.client
    id = str(uuid.uuid4()).replace("-", "")[0:8]
    playlist_name = f"Suggested Songs - {id}"
    playlist_description = "Playlist generated by the mood of the user - Happy"
    playlist = client.user_playlist_create(
        user=client.current_user()["id"],
        name=playlist_name,
        public=False,
        description=playlist_description,
    )

    track_ids = [f"spotify:track:{id}" for id in body["ids"]]
    client.playlist_add_items(playlist["id"], track_ids)
    return {"playlist": playlist["external_urls"]["spotify"]}


@app.get("/login")
def spotify_login():
    # RedirectResponse(app.sp.get_authorize_url())
    return FileResponse("static/login.html")


@app.get("/callback")
def spotify_callback(request: Request):
    return FileResponse("static/recent.html")


@app.get("/suggest")
def suggest():
    return FileResponse("static/suggest.html")


@app.get("/")
def redirect_to_login():
    return RedirectResponse("/login")


def main():
    uvicorn.run(app, host="127.0.0.1", port=5000)


if __name__ == "__main__":
    main()
