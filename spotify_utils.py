import json
from dataclasses import dataclass, field
import requests
import spotipy
import webbrowser

@dataclass
class ConnectSpotifyItem:
    '''
    Automatically sets the details of the connection to the Spotify API.
    '''
    username: str = field(default="koaladan43")
    client_id: str = field(default="a063191c16264c52ae9c31d237cd1206")
    client_secret: str = field(default="2a81399b653f4fd891b1d29da740f08c")
    redirect_uri: str = field(default="http://localhost:8888/callback/")
    def __post_init__(self):
        self.oauth_object = spotipy.SpotifyOAuth(self.client_id, self.client_secret, self.redirect_uri)
        self.token_dict = self.oauth_object.get_access_token()
        self.token = self.token_dict['access_token']

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

        self.artist = self.results["tracks"]["items"][0]["artists"][0]["name"]
        self.album = self.results["tracks"]["items"][0]["album"]["name"]
        self.album_image_link = self.results["tracks"]["items"][0]["album"]["images"][0]["url"]
        self.track_url = self.results["tracks"]["items"][0]["external_urls"]["spotify"]
        self.track_id = self.results["tracks"]["items"][0]["id"]

    def display_song_data(self) -> None:
        '''
        Display song data attributes such as artist, album, ... track_url.
        :return: None
        '''
        print("Artist:\t ", self.artist)
        print("Album:\t ", self.album)
        print("Album Image Link:\t ", self.album_image_link)
        print("Track ID: \t ", self.track_id)
        print("Track Link:\t ", self.track_url)

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

# if __name__ == "__main__":
#     spotify_creds = ConnectSpotifyItem()
#     song = Song(spotify_creds.token, "My Immortal")
#     song.display_song_data()
#     audio_data = song.get_audio_analysis()
#     print(audio_data)
# cats test