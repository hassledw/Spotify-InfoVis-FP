import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

### DATASET PROCESSING ###
old_df = pd.read_csv("./spotify.csv")
nan_values = old_df.isnull().sum()

df = pd.read_csv("./spotify.csv").dropna().drop(['Unnamed: 0'], axis=1)

# ### OUTLIER DETECTION & REMOVAL SECTION ###
# fig = plt.figure(figsize = (10,10))
# features = ["popularity", 'duration_ms', 'danceability', "energy", "loudness", 'speechiness', "acousticness", 'instrumentalness', 'liveness', 'valence', 'tempo', "track_genre"]
# top10_pop = df[[feature for feature in features]].groupby(
#     'track_genre').mean().reset_index().sort_values(by='popularity', ascending=False).head(10)
# top10_pop_with_songs = pd.DataFrame()
# for genre in top10_pop.track_genre.unique():
#     top10_pop_with_songs = pd.concat((
#         df[df['track_genre'] == genre], top10_pop_with_songs
#     ))
#
# ### PAIR PLOT ###
# pairplot = sns.pairplot(
#     top10_pop_with_songs[["popularity", 'duration_ms', 'danceability', "energy", "loudness", 'speechiness', "acousticness", 'instrumentalness', 'liveness', 'valence', 'tempo', "track_genre"]],
#     palette="husl",
#     hue="track_genre")
# plt.suptitle("Pair Plot")
# plt.show()
#
# ### BOX PLOT ###
# fig, axes = plt.subplots(3, 3, figsize=(18, 10))
# fig.suptitle('Box Plot of Features')
# sns.boxplot(ax=axes[0, 0], data=top10_pop_with_songs, x='track_genre', y='tempo')
# sns.boxplot(ax=axes[0, 1], data=top10_pop_with_songs, x='track_genre', y='danceability')
# sns.boxplot(ax=axes[0, 2], data=top10_pop_with_songs, x='track_genre', y='energy')
# sns.boxplot(ax=axes[1, 0], data=top10_pop_with_songs, x='track_genre', y='loudness')
# sns.boxplot(ax=axes[1, 1], data=top10_pop_with_songs, x='track_genre', y='speechiness')
# sns.boxplot(ax=axes[1, 2], data=top10_pop_with_songs, x='track_genre', y='acousticness')
# sns.boxplot(ax=axes[2, 0], data=top10_pop_with_songs, x='track_genre', y='instrumentalness')
# sns.boxplot(ax=axes[2, 1], data=top10_pop_with_songs, x='track_genre', y='liveness')
# sns.boxplot(ax=axes[2, 2], data=top10_pop_with_songs, x='track_genre', y='valence')
# plt.show()

### REMOVE OUTLIERS ###
numeric_df = df[["popularity", "duration_ms", "danceability", "energy", "loudness",
                   "speechiness", "acousticness", "instrumentalness", "liveness",
                   "valence", "tempo"]]
no_outliers = numeric_df[(np.abs(stats.zscore(numeric_df)) < 3).all(axis=1)]
print('Original Dataset w/ Outliers:')
print(df)
print(' ')
print('New Dataset w/o Outliers:')
print(no_outliers)

### PCA ###
X = StandardScaler().fit_transform(no_outliers)

pca = PCA(n_components='mle', svd_solver='full')
pca.fit(X)
X_pca = pca.transform(X)
print('Original Feature Space:', X.shape)
print('Transformed Feature Space:', X_pca.shape)
print('Explained variance ratio:', (pca.explained_variance_ratio_.cumsum().round(2)))

new_df = pd.DataFrame(data = X_pca)
print(' ')
print(f'Transformed dataset:')
print(new_df)

new_singular_values = pca.singular_values_
new_cond_num = pca.singular_values_[0] / pca.singular_values_[-1]
print(f'singular values of reduced dim: {new_singular_values}')
print(f'conditional number of reduced dim: {new_cond_num:.2f}')

### CUMULATIVE EXPLAINED VARIANCE PLOT ###
plt.plot(np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1, 1),
         np.cumsum(pca.explained_variance_ratio_ * 100))
plt.axhline(y=99, color="red", linestyle="--", linewidth=2)
plt.axvline(x=10, color="black", linestyle="--", linewidth=2)
plt.xticks(np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1, 1))
plt.grid()
plt.title("Cumulative Explained Variance on Number of Components", fontdict={'fontsize':15})
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance")
plt.show()

### HISTOGRAM ###
fig = plt.figure(figsize = (20, 20))
dist_types = ["norm"]
count = 1
for feature in no_outliers.columns:
    for i, dist in enumerate(dist_types):
        ax = fig.add_subplot(3, 4, count)
        params = ()
        stats.probplot(no_outliers[feature], dist=dist, sparams=params, plot=plt)
        ax.set_title(f"{feature} on {dist} plot", fontsize=15)
        count += 1
plt.suptitle("Distribution Evaluation for Numerical Features", fontsize=30)
plt.show()

### HEATMAP & CORRELATION MATRIX ###
plt.figure(figsize=(10, 10))
heatmap = sns.heatmap(no_outliers.corr(method="pearson"), vmin=-1.0,
                      vmax=1.0, annot=True, fmt=".1f", cmap="YlGnBu", cbar=True)
heatmap.set_title('Correlation Heatmap of Spotify Data w/o Outliers', fontdict={'fontsize':15}, pad=14);
plt.tight_layout()
plt.show()

#%%
import pandas as pd
from prettytable import PrettyTable

df = pd.read_csv("./spotify.csv").dropna().drop(['Unnamed: 0'], axis=1)
features = ["popularity", 'duration_ms', 'danceability', "energy", "loudness", 'speechiness', "acousticness", 'instrumentalness', 'liveness', 'valence', 'tempo', "track_genre"]
top10_pop = df[[feature for feature in features]].groupby(
    'track_genre').mean().reset_index().sort_values(by='popularity', ascending=False).head(10)
top10_pop_with_songs = pd.DataFrame()
for genre in top10_pop.track_genre.unique():
    top10_pop_with_songs = pd.concat((
        df[df['track_genre'] == genre], top10_pop_with_songs
    ))


def mean_value_comparison():
    # data
    pop_data = top10_pop_with_songs[top10_pop_with_songs['track_genre'] == 'pop']
    pop_dance = round(pop_data[['danceability']].mean().iat[0], 2)
    sertanejo_data = top10_pop_with_songs[top10_pop_with_songs['track_genre'] == 'sertanejo']
    sertanejo_dance = round(sertanejo_data[['danceability']].mean().iat[0], 2)
    emo_data = top10_pop_with_songs[top10_pop_with_songs['track_genre'] == 'emo']
    emo_dance = round(emo_data[['danceability']].mean().iat[0], 2)
    anime_data = top10_pop_with_songs[top10_pop_with_songs['track_genre'] == 'anime']
    anime_dance = round(anime_data[['danceability']].mean().iat[0], 2)
    indian_data = top10_pop_with_songs[top10_pop_with_songs['track_genre'] == 'indian']
    indian_dance = round(indian_data[['danceability']].mean().iat[0], 2)
    grunge_data = top10_pop_with_songs[top10_pop_with_songs['track_genre'] == 'grunge']
    grunge_dance = round(grunge_data[['danceability']].mean().iat[0], 2)
    sad_data = top10_pop_with_songs[top10_pop_with_songs['track_genre'] == 'sad']
    sad_dance = round(sad_data[['danceability']].mean().iat[0], 2)
    chill_data = top10_pop_with_songs[top10_pop_with_songs['track_genre'] == 'chill']
    chill_dance = round(chill_data[['danceability']].mean().iat[0], 2)
    kpop_data = top10_pop_with_songs[top10_pop_with_songs['track_genre'] == 'k-pop']
    kpop_dance = round(kpop_data[['danceability']].mean().iat[0], 2)
    popfilm_data = top10_pop_with_songs[top10_pop_with_songs['track_genre'] == 'pop-film']
    popfilm_dance = round(popfilm_data[['danceability']].mean().iat[0], 2)

    pop_energy = round(pop_data[['energy']].mean().iat[0], 2)
    sertanejo_energy = round(sertanejo_data[['energy']].mean().iat[0], 2)
    emo_energy = round(emo_data[['energy']].mean().iat[0], 2)
    anime_energy = round(anime_data[['energy']].mean().iat[0], 2)
    indian_energy = round(indian_data[['energy']].mean().iat[0], 2)
    grunge_energy = round(grunge_data[['energy']].mean().iat[0], 2)
    sad_energy = round(sad_data[['energy']].mean().iat[0], 2)
    chill_energy = round(chill_data[['energy']].mean().iat[0], 2)
    kpop_energy = round(kpop_data[['energy']].mean().iat[0], 2)
    popfilm_energy = round(popfilm_data[['energy']].mean().iat[0], 2)

    pop_loudness = round(pop_data[['loudness']].mean().iat[0], 2)
    sertanejo_loudness = round(sertanejo_data[['loudness']].mean().iat[0], 2)
    emo_loudness = round(emo_data[['loudness']].mean().iat[0], 2)
    anime_loudness = round(anime_data[['loudness']].mean().iat[0], 2)
    indian_loudness = round(indian_data[['loudness']].mean().iat[0], 2)
    grunge_loudness = round(grunge_data[['loudness']].mean().iat[0], 2)
    sad_loudness = round(sad_data[['loudness']].mean().iat[0], 2)
    chill_loudness = round(chill_data[['loudness']].mean().iat[0], 2)
    kpop_loudness = round(kpop_data[['loudness']].mean().iat[0], 2)
    popfilm_loudness = round(popfilm_data[['loudness']].mean().iat[0], 2)

    pop_tempo = round(pop_data[['tempo']].mean().iat[0], 2)
    sertanejo_tempo = round(sertanejo_data[['tempo']].mean().iat[0], 2)
    emo_tempo = round(emo_data[['tempo']].mean().iat[0], 2)
    anime_tempo = round(anime_data[['tempo']].mean().iat[0], 2)
    indian_tempo = round(indian_data[['tempo']].mean().iat[0], 2)
    grunge_tempo = round(grunge_data[['tempo']].mean().iat[0], 2)
    sad_tempo = round(sad_data[['tempo']].mean().iat[0], 2)
    chill_tempo = round(chill_data[['tempo']].mean().iat[0], 2)
    kpop_tempo = round(kpop_data[['tempo']].mean().iat[0], 2)
    popfilm_tempos = round(popfilm_data[['tempo']].mean().iat[0], 2)

    pop_liveness = round(pop_data[['liveness']].mean().iat[0], 2)
    sertanejo_liveness = round(sertanejo_data[['liveness']].mean().iat[0], 2)
    emo_liveness = round(emo_data[['liveness']].mean().iat[0], 2)
    anime_liveness = round(anime_data[['liveness']].mean().iat[0], 2)
    indian_liveness = round(indian_data[['liveness']].mean().iat[0], 2)
    grunge_liveness = round(grunge_data[['liveness']].mean().iat[0], 2)
    sad_liveness = round(sad_data[['liveness']].mean().iat[0], 2)
    chill_liveness = round(chill_data[['liveness']].mean().iat[0], 2)
    kpop_liveness = round(kpop_data[['liveness']].mean().iat[0], 2)
    popfilm_liveness = round(popfilm_data[['liveness']].mean().iat[0], 2)

    pop_dur = round(pop_data[['duration_ms']].mean().iat[0], 2)
    sertanejo_dur = round(sertanejo_data[['duration_ms']].mean().iat[0], 2)
    emo_dur = round(emo_data[['duration_ms']].mean().iat[0], 2)
    anime_dur = round(anime_data[['duration_ms']].mean().iat[0], 2)
    indian_dur = round(indian_data[['duration_ms']].mean().iat[0], 2)
    grunge_dur = round(grunge_data[['duration_ms']].mean().iat[0], 2)
    sad_dur = round(sad_data[['duration_ms']].mean().iat[0], 2)
    chill_dur = round(chill_data[['duration_ms']].mean().iat[0], 2)
    kpop_dur = round(kpop_data[['duration_ms']].mean().iat[0], 2)
    popfilm_dur = round(popfilm_data[['duration_ms']].mean().iat[0], 2)

    # table
    table = PrettyTable(['Feature/Genre', 'Danceability', 'Energy', 'Loudness', 'Tempo', 'Liveness', 'Duration'])
    table.add_row(['Pop', pop_dance, pop_energy, pop_loudness, pop_tempo, pop_liveness, pop_dur])
    table.add_row(['Sertanejo', sertanejo_dance, sertanejo_energy, sertanejo_loudness, sertanejo_tempo, sertanejo_liveness, sertanejo_dur])
    table.add_row(['Emo', emo_dance, emo_energy, emo_loudness, emo_tempo, emo_liveness, emo_dur])
    table.add_row(['Anime', anime_dance, anime_energy, anime_loudness, anime_tempo, anime_liveness, anime_dur])
    table.add_row(['Indian', indian_dance, indian_energy, indian_loudness, indian_tempo, indian_liveness, indian_dur])
    table.add_row(['Grunge', grunge_dance, grunge_energy, grunge_loudness, grunge_tempo, grunge_liveness, grunge_dur])
    table.add_row(['Sad', sad_dance, sad_energy, sad_loudness, sad_tempo, sad_liveness, sad_dur])
    table.add_row(['Chill', chill_dance, chill_energy, chill_loudness, chill_tempo, chill_liveness, chill_dur])
    table.add_row(['K-pop', kpop_dance, kpop_energy, kpop_loudness, kpop_tempo, kpop_liveness, kpop_dur])
    table.add_row(['Pop-film', popfilm_dance, popfilm_energy, popfilm_loudness, popfilm_tempos, popfilm_liveness, popfilm_dur])

    table.title = 'Mean Value Comparison for various features of the top 10 most popular genres'
    print(table)
mean_value_comparison()

