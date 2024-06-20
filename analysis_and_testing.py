import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("./spotify.csv").dropna()

X = StandardScaler().fit_transform(df.select_dtypes(include=np.number))

pca = PCA(n_components='mle', svd_solver='full')
pca.fit(X)
X_pca = pca.transform(X)
print('Original Feature Space:', X.shape)
print('Transformed Feature Space:', X_pca.shape)

new_df = pd.DataFrame(data = X_pca)
print(' ')
print(f'Transformed dataset:')
print(new_df.head())