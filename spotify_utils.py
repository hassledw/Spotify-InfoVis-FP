import json
import spotipy
import webbrowser

class Song:
    '''
    A Song class that acts like a Spotify API wrapper for easier querying.
    '''
    def __init__(self, spotify, songname):
        '''
        Populates song information attributes via Spotify API.

        :param songname: the name of the song.
        :param spotify: the Spotify API object.
        '''
        self.songname = songname
        results = spotify.search(self.songname, 1, 0, "track")

        self.artist = results["tracks"]["items"][0]["artists"][0]["name"]
        self.album = results["tracks"]["items"][0]["album"]["name"]
        self.album_image_link = results["tracks"]["items"][0]["album"]["images"][0]["url"]
        self.track_url = results["tracks"]["items"][0]["external_urls"]["spotify"]

    def display_song_data(self) -> None:
        '''
        Display song data attributes such as artist, album, ... track_url.
        :return: None
        '''
        print("Artist:\t ", self.artist)
        print("Album:\t ", self.album)
        print("Album Image Link:\t ", self.album_image_link)
        print("Track Link:\t ", self.track_url)

    def play_song(self) -> None:
        '''
        Plays a song on current device.
        :return: None
        '''
        webbrowser.open(self.track_url)



if __name__ == "__main__":
    USERNAME="koaladan43"
    CLIENT_ID="a063191c16264c52ae9c31d237cd1206"
    CLIENT_SECRET="2a81399b653f4fd891b1d29da740f08c"
    REDIRECT_URI="http://localhost:8888/callback/"

    oauth_object = spotipy.SpotifyOAuth(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI)
    token_dict = oauth_object.get_access_token()
    token = token_dict['access_token']
    spotify = spotipy.Spotify(auth=token)

    song = Song("My Immortal", spotify)
    song.display_song_data()
    song.play_song()