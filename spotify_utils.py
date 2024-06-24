import json
from dataclasses import dataclass, field
import requests
import spotipy
import webbrowser
import requests
import base64


@dataclass
class ConnectSpotifyItem:
    '''
    Automatically sets the details of the connection to the Spotify API.

    Used with USER PREFERNCES.
    '''
    username: str = field(default="koaladan43")
    client_id: str = field(default="a063191c16264c52ae9c31d237cd1206")
    client_secret: str = field(default="2a81399b653f4fd891b1d29da740f08c")
    redirect_uri: str = field(default="http://localhost:8888/callback/")
    def __post_init__(self):
        self.oauth_object = spotipy.SpotifyOAuth(self.client_id, self.client_secret, self.redirect_uri)
        self.token_dict = self.oauth_object.get_access_token()
        self.token = self.token_dict['access_token']

def get_access_token():
    '''
    Gets the access token from the Spotify API, NO USER ID.
    Source: https://developer.spotify.com/documentation/spotify/api/v1/
    :return: access token as a string.
    '''
    client_id = "a063191c16264c52ae9c31d237cd1206"
    client_secret = "2a81399b653f4fd891b1d29da740f08c"
    client_base64 = base64.b64encode(f"{client_id}:{client_secret}".encode())

    url = "https://accounts.spotify.com/api/token"
    data = {
        "grant_type": "client_credentials"
    }
    headers = {
        "Authorization": f"Basic {client_base64.decode()}"
    }

    request = requests.post(url, data=data, headers=headers)
    response_data = request.json()

    token = response_data['access_token']

    return token

class Song:
    '''
    A Song class that acts like a Spotify API wrapper for easier querying.
    '''
    def __init__(self, token, songname):
        '''
        Populates song information attributes via Spotify API.

        :param songname: the name of the song.
        :param spotify: the Spotify API object.
        '''
        self.songname = songname
        self.token = token
        self.spotify = spotipy.Spotify(auth=self.token)
        self.results = self.spotify.search(self.songname, 1, 0, "track")

        self.track = self.results["tracks"]["items"][0]

        self.artist = self.track["artists"][0]["name"]
        self.songname = self.track["name"]
        self.album = self.track["album"]["name"]
        self.album_image_link = self.track["album"]["images"][0]["url"]
        self.track_url = self.track["external_urls"]["spotify"]
        self.track_id = self.track["id"]

    def display_song_data(self) -> str:
        '''
        Display song data attributes such as artist, album, ... track_url.
        :return: None
        '''
        print("Artist:\t ", self.artist)
        print("Track:\t ", self.songname)
        print("Album:\t ", self.album)
        print("Album Image Link:\t ", self.album_image_link)
        print("Track ID: \t ", self.track_id)
        print("Track Link:\t ", self.track_url)

        info = f"""
        Artist:\t {self.artist}
        Track:\t {self.songname}
        Album:\t {self.album}
        Track ID:\t {self.track_id}
        Track Link:\t {self.track_url}
        """

        return info

    def play_song(self) -> None:
        '''
        Plays a song on current device.
        :return: None
        '''
        webbrowser.open(self.track_url)

    def get_audio_analysis(self) -> dict:
        '''
        Gets all the waveform and audio data necessary to plot
        song.

        :return: JSON response of song audio representation.
        '''
        audio_data_url = f'https://api.spotify.com/v1/audio-analysis/{self.track_id}'
        response = requests.get(audio_data_url, headers={
            'Authorization': f'Bearer {self.token}'
        })
        data = response.json()
        return data

    def create_dataset_entry(self) -> dict:
        '''
        Creates a Song entry for our Spotify dataset. This will allow
        us to add more data to our dataset if needed.

        :return: a dictionary of all the values.
        '''
        track_audio_features = self.spotify.audio_features(self.track_id)[0]

        return {
            "track_id": self.track_id,
            "artists": self.artist,
            "album_name": self.album,
            "track_name": self.songname,
            "explicit": self.track["explicit"],
            "popularity": self.track["popularity"],
            "key": track_audio_features["key"],
            "mode": track_audio_features["mode"],
            "time_signature": track_audio_features["time_signature"],
            "duration_ms": track_audio_features["duration_ms"],
            "danceability": track_audio_features["danceability"],
            "energy": track_audio_features["energy"],
            "loudness": track_audio_features["loudness"],
            "speechiness": track_audio_features["speechiness"],
            "acousticness": track_audio_features["acousticness"],
            "instrumentalness": track_audio_features["instrumentalness"],
            "liveness": track_audio_features["liveness"],
            "valence": track_audio_features["valence"],
            "tempo": track_audio_features["tempo"]
        }


if __name__ == "__main__":
    #     spotify_creds = ConnectSpotifyItem()

    r = requests.post(token_url, data=token_data, headers=token_headers)
    token_response_data = r.json()

    access_token = token_response_data['access_token']

    print("Access Token:", access_token)
    song = Song(access_token, "My Immortal")
    song.display_song_data()
    # audio_data = song.get_audio_analysis()
    dataset_entry = song.create_dataset_entry()

    print(dataset_entry)
