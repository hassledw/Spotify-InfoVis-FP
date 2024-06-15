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
STACKED BAR PLOT (features made up by danceability, energy, speechiness, acousticness, instrumentalness, liveness, and valence
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
PIE CHART: Percentage of each key used throughout
'''
zero_df = df[df['key']==0]
zero_df_p = round((len(zero_df) / len(df)) * 100, 2)
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
eleven_df = df[df['key']==11]
eleven_df_p = round((len(eleven_df) / len(df)) * 100, 2)
all_p = np.array([zero_df_p, one_df_p, two_df_p, three_df_p, four_df_p, five_df_p, six_df_p, seven_df_p, eight_df_p, nine_df_p, ten_df_p, eleven_df_p])
p_labels = ['Key 0', 'Key 1', 'Key 2', 'Key 3', 'Key 4', 'Key 5', 'Key 6', 'Key 7', 'Key 8', 'Key 9', 'Key 10', 'Key 11']
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
songdata, sr = librosa.load("songs/DontStopBelievinJourney.mp3")

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

'''
Cluster Map: explore relationships between features
'''
# data
features1 = ['popularity', 'energy', 'danceability', 'liveness']
top10_energy = df[[feature for feature in features1]].groupby(
    'energy').mean().reset_index().sort_values(by='popularity', ascending=False).head(10)
# map
cluster = sns.clustermap(top10_energy, metric='correlation', standard_scale = 1,
                         cmap='viridis', figsize=(10, 10))
cluster.fig.subplots_adjust(right=0.7)
cluster.ax_cbar.set_position((0.8, .2, .03, .4))
cluster.fig.suptitle('Cluster Map of Spotify Data', fontsize=25)
plt.show()

'''
AREA PLOT: Tempo vs Valence
'''
areadf = df[["artists", "track_name",
             "valence", "tempo", "energy", "danceability"]].copy()

tempo_speed_categories = [0, 90, 140, 160, 200]
tempo_speed_names = ["Slow", "Relaxed", "Medium", "Fast", "Extra Fast"]
name_count = 0

# init all TempoCategory values to None.
areadf.loc[:, 'TempoCategory'] = "None"

for i in range(0, len(tempo_speed_categories)):
    begin_tempo = tempo_speed_categories[i]
    end_tempo = None
    tempo_cond = None

    if i == len(tempo_speed_categories) - 1:
        tempo_cond = (areadf["tempo"] >= begin_tempo)
    else:
        end_tempo = tempo_speed_categories[i + 1]
        tempo_cond = (areadf["tempo"] >= begin_tempo) & (areadf["tempo"] <= end_tempo)

    areadf.loc[tempo_cond, 'TempoCategory'] = tempo_speed_names[name_count]
    name_count+=1

areadf["TempoCategory"] = pd.Categorical(areadf["TempoCategory"], categories=tempo_speed_names, ordered=True)
areadf = areadf.groupby("TempoCategory").agg(
    {"valence": "mean", "energy": "mean", "danceability": "mean"}).sort_values(by="TempoCategory")

#print(areadf.head())

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
areadf.plot(kind='area', alpha=0.5, color=["purple", "blue", "red"], ax=ax)
ax.set_title("Average Valence, Energy, and Danceability by Tempo Categories")
ax.set_xlabel("Tempo Category")
ax.set_ylabel("Average Value")
plt.show()


'''
Hexbin Plot: Energy vs Liveness
'''
sns.jointplot(df, x="energy", y="liveness", kind="hex", color="green")
plt.suptitle("Energy vs Liveness on all Songs")
plt.show()

'''
Strip Plot: Showing Explicit values in hip-hop Genre
'''
# hip hop genre
hiphopdf = df[df["track_genre"] == "hip-hop"]

# top 10 most represented artists in hip-hop genre
sortedhipdf = hiphopdf["artists"].value_counts().head(10).reset_index()

# get all songs by the top 10 artists
hiphopresdf = pd.DataFrame(columns=hiphopdf.columns.tolist())

for artist in sortedhipdf['artists']:
    hiphopresdf = pd.concat([hiphopresdf, hiphopdf[hiphopdf["artists"] == artist]])

# print(hiphopresdf.shape)
# print(hiphopresdf.head())

fig = plt.figure(figsize=(10, 8))
sns.stripplot(data=hiphopresdf, x="energy", y="artists", hue="explicit")
plt.suptitle("Explicit distribution on Hip-hop Genre")
plt.show()

'''
Swarm Plot: Different Modes and Valence on Grunge genre
'''
fig = plt.figure(figsize=(10, 8))
grungedf = df[df["track_genre"] == "grunge"].copy()
grungedf["mode"] = pd.Categorical.from_codes(grungedf['mode'], ['Minor', 'Major'])

custom_palette = {
    "Minor": "salmon",
    "Major": "lightgreen"
}
sns.swarmplot(data=grungedf, x="mode", y="valence", hue="mode", palette=custom_palette, legend=True)

plt.suptitle("Different Modes vs Valence on Grunge Genre", fontsize=20)
plt.show()

'''
Violin Plot: Genre vs Danceability on top 10 most represented genres.
'''
fig = plt.figure(figsize=(10, 10))

# get top 10 most represented genres
top10repgenres = df["track_genre"].value_counts().head(10).reset_index()

# populate a dataframe for these genres.
violindf = pd.DataFrame(columns=df.columns.tolist())

for genre in top10repgenres["track_genre"]:
    violindf = pd.concat([violindf, df[df["track_genre"] == genre]])

sns.violinplot(violindf, x="danceability", y="track_genre", hue="track_genre", palette="viridis")
plt.suptitle("Danceability Distribution on Top-10 Selected Genres", fontsize=20)
plt.show()

'''
KDE Plot: Energy density in each time signature
'''
sns.kdeplot(
   data=df, x="energy", hue="time_signature",
   fill=True, common_norm=False, palette="crest",
   alpha=0.6, linewidth=0)
plt.title("Energy density in each time signature")
plt.show()

'''
Rug Plot: scatter plot and rug plots for the top 10 genre liveness, loudness, energy, and valence features
'''
# data
new_features = ["popularity", "energy", "loudness", "acousticness", 'speechiness', 'liveness', 'valence', 'time_signature', "track_genre"]
new_top10_pop = df[[feature for feature in new_features]].groupby(
   'track_genre').mean().reset_index().sort_values(by='popularity', ascending=False).head(10)
# plots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
fig.suptitle('scatter plot and rug plots for the top 10 genre features')
sns.scatterplot(ax=axes[0, 0], data=new_top10_pop, x="popularity", y="liveness", hue="track_genre")
sns.rugplot(ax=axes[0, 0], data=new_top10_pop, x="popularity", y="liveness", hue="track_genre", height=.05)
sns.scatterplot(ax=axes[0, 1], data=new_top10_pop, x="popularity", y="loudness", hue="track_genre")
sns.rugplot(ax=axes[0, 1], data=new_top10_pop, x="popularity", y="loudness", hue="track_genre", height=.05)
sns.scatterplot(ax=axes[1, 0], data=new_top10_pop, x="popularity", y="energy", hue="track_genre")
sns.rugplot(ax=axes[1, 0], data=new_top10_pop, x="popularity", y="energy", hue="track_genre", height=.05)
sns.scatterplot(ax=axes[1, 1], data=new_top10_pop, x="popularity", y="valence", hue="track_genre")
sns.rugplot(ax=axes[1, 1], data=new_top10_pop, x="popularity", y="valence", hue="track_genre", height=.05)
plt.show()

'''
Boxen Plot: tempo distribution
'''
ax = sns.boxenplot(x=df["tempo"])
ax.set_title('Tempo distribution')
plt.show()

'''
Joint Plot w KDE: tempo and danceability for each time signature
'''
fig = plt.figure(figsize=(10, 20))
joint = sns.jointplot(data=df, x="tempo", y="danceability", hue="time_signature", kind="kde")
joint.fig.suptitle('tempo and danceability for each time signature')
joint.fig.subplots_adjust(top=0.9)
plt.show()
