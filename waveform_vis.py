import pandas as pd
from spotify_utils import ConnectSpotifyItem, Song
import numpy as np
import plotly.express as px

def loudness_to_amplitude(loudness):
    '''
    Converts loudness (in dB) to amplitude

    :param loudness: loudness (in dB)
    :return: amplitude
    '''
    return np.power(10, np.divide(loudness, 20))
def plot_song_waveform(song : Song, audio_data):
    '''
    Extracts the segments form the song and other data like loudness
    to compute and visualize the waveform (time domain).

    :param song: the song object
    :return: None
    '''
    segments = audio_data['segments']
    loudness = np.array([segment['loudness_max'] for segment in segments])
    times = np.array([segment['start'] for segment in segments])
    amplitude = loudness_to_amplitude(loudness)

    df = pd.DataFrame({'time': times, 'amplitude': amplitude})

    fig = px.line(df,
                  x='time',
                  y='amplitude',
                  title=f'Waveform of \"{song.songname}\" by {song.artist}')

    fig.update_layout(
        title_font_color="blue",
        title_font_size=25,
        title={'y': .95, 'x': 0.5},
        font_size=20,
        xaxis_title="Time",
        xaxis_title_font_color="black",
        yaxis_title="Amplitude (dB)",
        yaxis_title_font_color="black",
    )

    fig.update_traces(
        line=dict(color="darkgreen", width=2)
    )

    return fig

if __name__ == "__main__":
    spotify_creds = ConnectSpotifyItem()
    song = Song(spotify_creds.token, songname="My heart will go on")
    song.display_song_data()
    audio_data = song.get_audio_analysis()
    plot_song_waveform(song, audio_data)