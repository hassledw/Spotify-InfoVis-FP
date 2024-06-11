# phase 1 graphs
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import librosa

df = pd.read_csv("./spotify.csv").dropna()
genre = df['track_genre'].unique()
genre_list = df['track_genre'].unique().tolist()
numeric_df = df[["popularity", "duration_ms", "danceability", "energy", "loudness",
                   "speechiness", "acousticness", "instrumentalness", "liveness",
                   "valence", "tempo"]]

print("Before cleaning: ", df.shape)
df = df.drop(['track_id'], axis=1)
df = df.dropna()
print("After cleaning: ", df.shape)

'''
LINE PLOT: Danceability + energy means by genre.
'''
means_by_genre = df.groupby('track_genre').mean(numeric_only=True)
genre_energy = means_by_genre[['energy']]
genre_danceability = means_by_genre[['danceability']]
genre_valence = means_by_genre[['valence']]
plt.figure(figsize = (20, 15))
plt.plot(genre_energy, linewidth=1, color= 'red', label = 'Energy')
plt.plot(genre_valence, linewidth=1, color= 'blue', label = 'Valence')
plt.plot(genre_danceability, linewidth=1, color= 'orange', label = 'Danceability')
plt.xticks(genre, rotation = 90, fontsize = 10)
plt.xlabel('Genre', fontsize = 20)
plt.ylabel('Mean Amount', fontsize = 20)
plt.title('Energy, Valence, and Danceability Means for each Genre', fontsize = 35)
plt.legend(loc = 'upper right')
plt.grid()
plt.show()

'''
GROUPED BAR PLOT: popularity and tempo
'''
vector = np.vectorize(np.int_)
genre_popularity = means_by_genre['popularity']
pop_bars = vector(genre_popularity)
genre_tempo = means_by_genre['tempo']
temp_bars = vector(genre_tempo)
num_bars = np.arange(len(genre_list))
plt.figure(figsize = (20, 15))
plt.bar(num_bars-0.2, pop_bars, width = 0.4, align = 'center', color= 'blue', label = 'Popularity')
plt.bar(num_bars+0.2, temp_bars, width = 0.4, align = 'center', color= 'green', label = 'Tempo')
plt.xticks(num_bars, genre, rotation = 90, fontsize = 10)
plt.ylabel('Mean Amount', fontsize = 20)
plt.xlabel('Genre', fontsize = 20)
plt.title('Popularity and Tempo Means for each Genre', fontsize = 35)
plt.legend()
plt.show()

'''
STACKED BAR PLOT (features made up by danceability, energy, speechiess, acousticness, instrumentalness, liveness, and valence
by genre
'''
genre_danceability1 = means_by_genre['danceability']
dance_bars1 = vector(genre_danceability1)
genre_energy1 = means_by_genre['energy']
energy_bars1 = vector(genre_energy1)
genre_speech1 = means_by_genre['speechiness']
speech_bars1 = vector(genre_speech1)
genre_acoust1 = means_by_genre['acousticness']
acoust_bars1 = vector(genre_acoust1)
genre_inst1 = means_by_genre['instrumentalness']
inst_bars1 = vector(genre_inst1)
genre_live1 = means_by_genre['liveness']
live_bars1 = vector(genre_live1)
genre_val1 = means_by_genre['valence']
val_bars1 = vector(genre_val1)
plt.figure(figsize = (25, 18))
plt.bar(num_bars, dance_bars1, color = 'red', label = 'Danceability')
plt.bar(num_bars, genre_energy1, bottom = dance_bars1, color = 'blue', label = 'Energy')
plt.bar(num_bars, genre_speech1, bottom = dance_bars1 + genre_energy1, color = 'green', label = 'Speechiness')
plt.bar(num_bars, genre_acoust1, bottom = dance_bars1 + genre_energy1 + genre_speech1, color = 'orange', label = 'Acousticness')
plt.bar(num_bars, genre_inst1, bottom = dance_bars1 + genre_energy1 + genre_speech1 + genre_acoust1, color = 'yellow', label = 'Instrumentalness')
plt.bar(num_bars, genre_live1, bottom = dance_bars1 + genre_energy1 + genre_speech1 + genre_acoust1 + genre_inst1, color = 'purple', label = 'Liveness')
plt.bar(num_bars, genre_val1, bottom = dance_bars1 + genre_energy1 + genre_speech1 + genre_acoust1 + genre_inst1 + genre_live1, color = 'pink', label = 'Valence')
plt.xlabel('Genre', fontsize = 25)
plt.ylabel('Attribute Mean Amounts', fontsize = 25)
plt.title('Amount of Attributes by Genre', fontsize = 35)
plt.xticks(num_bars, genre, rotation = 90, fontsize = 10)
plt.legend(fontsize = 20)
plt.show()

'''
PIE CHART: Precentage of each key used throughout
'''
one_df = df[df['key']==1]
one_df_p = round((len(one_df) / len(df)) * 100, 2)
two_df = df[df['key']==2]
two_df_p = round((len(two_df) / len(df)) * 100, 2)
three_df = df[df['key']==3]
three_df_p = round((len(three_df) / len(df)) * 100, 2)
four_df = df[df['key']==4]
four_df_p = round((len(four_df) / len(df)) * 100, 2)
five_df = df[df['key']==5]
five_df_p = round((len(five_df) / len(df)) * 100, 2)
six_df = df[df['key']==6]
six_df_p = round((len(six_df) / len(df)) * 100, 2)
seven_df = df[df['key']==7]
seven_df_p = round((len(seven_df) / len(df)) * 100, 2)
eight_df = df[df['key']==8]
eight_df_p = round((len(eight_df) / len(df)) * 100, 2)
nine_df = df[df['key']==9]
nine_df_p = round((len(nine_df) / len(df)) * 100, 2)
ten_df = df[df['key']==10]
ten_df_p = round((len(ten_df) / len(df)) * 100, 2)
all_p = np.array([one_df_p, two_df_p, three_df_p, four_df_p, five_df_p, six_df_p, seven_df_p, eight_df_p, nine_df_p, ten_df_p])
p_labels = ['Key 1', 'Key 2', 'Key 3', 'Key 4', 'Key 5', 'Key 6', 'Key 7', 'Key 8', 'Key 9', 'Key 10']
plt.pie(all_p, labels = p_labels, autopct='%0.2f%%')
plt.title('Keys used in Spotify Songs')
plt.show()

'''
DISTRIBUTION PLOT: duration on songs
'''
durations = df['duration_ms']
dis = sns.displot(durations, kde=True)
dis.figure.set_figwidth((20))
dis.figure.set_figheight((10))
plt.xlim(0, 0.8*10**6)
plt.title('Duration Distribution')
plt.xlabel('Duration')
plt.ylabel('Count')
plt.show()


'''
HEATMAP (w/cbar): Depicts correlation matrix in heatmap.
'''
plt.figure(figsize=(10, 10))
heatmap = sns.heatmap(numeric_df.corr(method="pearson"), vmin=-1.0,
                      vmax=1.0, annot=True, fmt=".1f", cmap="YlGnBu", cbar=True)
heatmap.set_title('Correlation Heatmap of Numeric Spotify Data', fontdict={'fontsize':15}, pad=14);
plt.tight_layout()
plt.show()

'''
HORIZONTAL BAR PLOT: top 10 mean popularity by genre
'''
mean_pop = df[['popularity', 'track_genre']].groupby(
    'track_genre').mean().reset_index().sort_values(by='popularity', ascending=False).head(10)

plt.figure(figsize=(12, 8))
meanpopplot = sns.barplot(x='popularity', y='track_genre', data=mean_pop, hue="track_genre", palette="mako")
meanpopplot.set_title('Top 10 Mean Popularity by Genre', fontdict={'fontsize':15}, pad=14)
plt.xlabel('Mean Popularity')
plt.ylabel('Genre')

plt.tight_layout()
plt.show()

'''
COUNT PLOT: Most represented artists by popular genres
'''
fig = plt.figure(figsize=(30, 15))

colors = ["green", "orange", "red", "blue", "purple"]
for i, popular_genre in enumerate(mean_pop.track_genre.unique()):
    ax = fig.add_subplot(2, 5, i + 1)
    counts = df[df['track_genre'] == popular_genre][['artists', 'track_genre']].groupby(
        "track_genre").value_counts(sort=True, ascending=False).head(10).reset_index()

    sns.barplot(data=counts, x="artists", y="count", color=colors[i % 5], ax=ax)
    ax.set_title(f"Top 10 Most Represented Artists in {popular_genre}", fontdict={'fontsize':15}, pad=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)

plt.suptitle("Top-10 Represented Artists Across Popular Genres", fontsize=30)
plt.tight_layout()
plt.show()

'''
PAIR PLOT: Showing pair plot of correlated features
'''
fig = plt.figure(figsize = (10,10))
features = ["popularity", "energy", "loudness", "acousticness", "track_genre"]
top10_pop = df[[feature for feature in features]].groupby(
    'track_genre').mean().reset_index().sort_values(by='popularity', ascending=False).head(10)
print(top10_pop)

top10_pop_with_songs = pd.DataFrame()
for genre in top10_pop.track_genre.unique():
    top10_pop_with_songs = pd.concat((
        df[df['track_genre'] == genre], top10_pop_with_songs
    ))

pairplot = sns.pairplot(
    top10_pop_with_songs[["energy", "loudness", "acousticness", "track_genre"]],
    palette="husl",
    hue="track_genre")

# pairplot.fig.suptitle('Pair plot on Energy, Loudness, and Acousticness', y=0.95)
plt.show()

'''
QQ PLOT: Statistical Distribution Evaluation on All Numeric Features.
'''
fig = plt.figure(figsize = (20, 20))

dist_types = ["norm", "beta", "expon", "lognorm"]
count = 1
for feature in numeric_df[["duration_ms", "danceability", "valence"]].columns:
    for i, dist in enumerate(dist_types):
        ax = fig.add_subplot(3, 4, count)
        params = ()
        if dist == "beta":
            params = (2, 5)
        elif dist == "expon":
            params = ()
        elif dist == "lognorm":
            params = (0.95)
        stats.probplot(numeric_df[feature], dist=dist, sparams=params, plot=plt)
        ax.set_title(f"{feature} on {dist} plot", fontsize=15)
        count += 1
plt.suptitle("Distribution Evaluation on Duration, Danceability, Valence", fontsize=30)
plt.show()

'''
3D PLOT + Contour: STFT of Don't Stop Believin'
'''

# get the amplitude->time domain signal and sampling rate of song.
songdata, sr = librosa.load("./DontStopBelievinJourney.mp3")

# create the STFT from the song to convert songdata from amp->time to freq->time.
stft = librosa.stft(songdata)

# converts the STFT amplitude to amplitude in DB, as humans hear loudness in log-scale.
# sets ref=np.max to make the max value referenced as 0.
A = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

# creates a time array the same size as amp and freq.
times = librosa.times_like(A, sr=sr)

# generates the frequency values corresponding to the rows of the STFT
frequencies = librosa.fft_frequencies(sr=sr)

T, F = np.meshgrid(times, frequencies)

fig = plt.figure(figsize=(10, 10))
grid = fig.add_gridspec(2, 3, height_ratios=[3, 1])
ax = fig.add_subplot(grid[0, :], projection='3d')
ax.set_title("3D Spectrogram of \"Don\'t Stop Believin'\" - Journey", fontsize=20, family='serif')
ax.set_xlabel('Time')
ax.set_ylabel('Frequency')
ax.set_zlabel('Amplitude')
ax.set_zlim(-100, 0)

ax.plot_surface(T, F, A, cmap='viridis', edgecolor='none', alpha=0.9, linewidth=0.5)
twod = ax.contour(T, F, A, zdir='z', offset=-100, cmap='coolwarm', alpha=0.5)
freqamp = ax.contour(T, F, A, zdir='x', offset=-40, cmap='coolwarm', alpha=0.5)
timeamp = ax.contour(T, F, A, zdir='y', offset=12000, cmap='coolwarm', alpha=0.5)

ax2 = fig.add_subplot(grid[1, 0])
ax2.contour(T, F, A, levels=twod.levels, cmap='coolwarm')
ax2.set_xlabel('Time')
ax2.set_ylabel('Frequency')

ax3 = fig.add_subplot(grid[1, 1])
ax3.contour(F, A, T, levels=freqamp.levels, cmap='coolwarm')
ax3.set_xlabel('Frequency')
ax3.set_ylabel('Amplitude')

ax4 = fig.add_subplot(grid[1, 2])
ax4.contour(T, A, F, levels=timeamp.levels, cmap='coolwarm')
ax4.set_xlabel('Time')
ax4.set_ylabel('Amplitude')
plt.tight_layout()
plt.show()
