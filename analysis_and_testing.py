import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

### DATASET PROCESSING ###
old_df = pd.read_csv("./spotify.csv")
nan_values = old_df.isnull().sum()
#print(old_df)

df = pd.read_csv("./spotify.csv").dropna().drop(['Unnamed: 0'], axis=1)
#print(df.head(5))

X = StandardScaler().fit_transform(df.select_dtypes(include=np.number))
# singular values and condition number for og
#singular_values = np.linalg.svd(X)
#cond_num = round(np.linalg.cond(X), 2)
#print(f'singular values of original dim: {singular_values}')
#print(f'conditional number of original dim: {cond_num}')

### OUTLIER DETECTION & REMOVAL SECTION ###
fig = plt.figure(figsize = (10,10))
features = ["popularity", 'duration_ms', 'danceability', "energy", "loudness", 'speechiness', "acousticness", 'instrumentalness', 'liveness', 'valence', 'tempo', "track_genre"]
top10_pop = df[[feature for feature in features]].groupby(
    'track_genre').mean().reset_index().sort_values(by='popularity', ascending=False).head(10)
top10_pop_with_songs = pd.DataFrame()
for genre in top10_pop.track_genre.unique():
    top10_pop_with_songs = pd.concat((
        df[df['track_genre'] == genre], top10_pop_with_songs
    ))

### PAIR PLOT ###
pairplot = sns.pairplot(
    top10_pop_with_songs[["popularity", 'duration_ms', 'danceability', "energy", "loudness", 'speechiness', "acousticness", 'instrumentalness', 'liveness', 'valence', 'tempo', "track_genre"]],
    palette="husl",
    hue="track_genre")
plt.suptitle("Pair Plot")
plt.show()

### BOX PLOT ###
fig, axes = plt.subplots(3, 3, figsize=(18, 10))
fig.suptitle('Box Plot of Features')
sns.boxplot(ax=axes[0, 0], data=top10_pop_with_songs, x='track_genre', y='tempo')
sns.boxplot(ax=axes[0, 1], data=top10_pop_with_songs, x='track_genre', y='danceability')
sns.boxplot(ax=axes[0, 2], data=top10_pop_with_songs, x='track_genre', y='energy')
sns.boxplot(ax=axes[1, 0], data=top10_pop_with_songs, x='track_genre', y='loudness')
sns.boxplot(ax=axes[1, 1], data=top10_pop_with_songs, x='track_genre', y='speechiness')
sns.boxplot(ax=axes[1, 2], data=top10_pop_with_songs, x='track_genre', y='acousticness')
sns.boxplot(ax=axes[2, 0], data=top10_pop_with_songs, x='track_genre', y='instrumentalness')
sns.boxplot(ax=axes[2, 1], data=top10_pop_with_songs, x='track_genre', y='liveness')
sns.boxplot(ax=axes[2, 2], data=top10_pop_with_songs, x='track_genre', y='valence')
plt.show()

### REMOVE OUTLIERS ###

### PCA ###
pca = PCA(n_components='mle', svd_solver='full')
pca.fit(X)
X_pca = pca.transform(X)
print('Original Feature Space:', X.shape)
print('Transformed Feature Space:', X_pca.shape)

new_df = pd.DataFrame(data = X_pca)
print(' ')
print(f'Transformed dataset:')
print(new_df.head())

