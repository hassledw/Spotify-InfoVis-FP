import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
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
no_outliers = df[(np.abs(stats.zscore(df.select_dtypes(include=np.number))) < 3).all(axis=1)]
print('Original Dataset w/ Outliers:')
print(df)
print(' ')
print('New Dataset w/o Outliers:')
print(no_outliers)

#%%
### PCA ###
X = StandardScaler().fit_transform(no_outliers.select_dtypes(include=np.number))

pca = PCA(n_components='mle', svd_solver='full')
pca.fit(X)
X_pca = pca.transform(X)
print('Original Feature Space:', X.shape)
print('Transformed Feature Space:', X_pca.shape)

new_df = pd.DataFrame(data = X_pca)
print(' ')
print(f'Transformed dataset:')
print(new_df)

new_singular_values = pca.singular_values_
new_cond_num = pca.singular_values_[0] / pca.singular_values_[-1]
print(f'singular values of reduced dim: {new_singular_values}')
print(f'conditional number of reduced dim: {new_cond_num:.2f}')

plt.plot(np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1, 1),
         np.cumsum(pca.explained_variance_ratio_ * 100))
plt.axhline(y=95, color="red", linestyle="--", linewidth=2)
plt.axvline(x=11.5, color="black", linestyle="--", linewidth=2)
plt.xticks(np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1, 1))
plt.grid()
plt.title("Cumulative Explained Variance on Number of Components", fontdict={'fontsize':15})
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance")
plt.show()
