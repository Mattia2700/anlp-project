import datetime
import os

import spotipy
from fastapi import FastAPI
from pydantic import BaseModel
from spotipy.oauth2 import SpotifyOAuth
from starlette.requests import Request
from starlette.responses import RedirectResponse

os.environ["SPOTIPY_CLIENT_ID"] = "f516c752459a4b94acba6a768cad9c43"
os.environ["SPOTIPY_CLIENT_SECRET"] = "641839e196f24feda3341dd5ac972749"
os.environ["SPOTIPY_REDIRECT_URI"] = "http://localhost:5000/callback"

app = FastAPI()
app.scope = "user-read-recently-played"
app.sp = SpotifyOAuth(scope=app.scope)
app.client = None
class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Foo",
                    "description": "A very nice Item",
                    "price": 35.4,
                    "tax": 3.2,
                }
            ]
        }
    }

@app.get("/callback")
def anything(request: Request):
    access_token = None
    code = app.sp.parse_response_code(request.query_params["code"])
    if code:
        token_info = app.sp.get_access_token(code)
        access_token = token_info["access_token"]

    if access_token:
        print(access_token)
        app.client = spotipy.Spotify(auth=access_token)
        last_week_timestamp = int((datetime.datetime.now() - datetime.timedelta(weeks=1)).timestamp() * 1000)

        results = app.client.current_user_recently_played(after=last_week_timestamp)

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

        return {"songs": actual_songs}


@app.get("/login")
def spotify_login():
    return RedirectResponse(app.sp.get_authorize_url())