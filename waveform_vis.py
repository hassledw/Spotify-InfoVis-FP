import matplotlib.pyplot as plt
from spotify_utils import ConnectSpotifyItem, Song
import numpy as np
import librosa

def loudness_to_amplitude(loudness):
    '''
    Converts loudness (in dB) to amplitude

    :param loudness: loudness (in dB)
    :return: amplitude
    '''
    return np.power(10, np.divide(loudness, 20))
def plot_song_waveform(song : Song) -> None:
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

    plt.figure(figsize=(15, 6))
    plt.plot(times, amplitude)
    plt.title(f'Waveform of {song.songname} by {song.artist}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.show()

if __name__ == "__main__":
    spotify_creds = ConnectSpotifyItem()
    song = Song(spotify_creds.token, songname="Summertime")
    song.display_song_data()
    audio_data = song.get_audio_analysis()
    plot_song_waveform(song)