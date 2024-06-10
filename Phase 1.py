# phase 1 graphs
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("./spotify.csv").dropna()
genre = df['track_genre'].unique()
genre_list = df['track_genre'].unique().tolist()
df.drop(['track_id'], axis=1)

# line plot (energy + dancebility means by genre)
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

#grouped bar plot (popularity and tempo grouped by genre)
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

# stacked/count? bar plot (features made up by danceability, energy, speechiess, acousticness, instrumentalness, liveness, and valence
# by genre
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
plt.bar(num_bars, genre_val1, bottom = dance_bars1 + genre_energy1 + genre_speech1 + genre_acoust1 + genre_inst1 + genre_live1, color = 'purple', label = 'Valence')
plt.xlabel('Genre', fontsize = 25)
plt.ylabel('Attribute Mean Amounts', fontsize = 25)
plt.title('Amount of Attributes by Genre', fontsize = 35)
plt.xticks(num_bars, genre, rotation = 90, fontsize = 10)
plt.legend(fontsize = 20)
plt.show()

# pie chart of the percentage of each key used throughout
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
plt.pie(all_p, labels = p_labels)
plt.title('Keys used in Spotify Songs')
plt.show()

# distribution plot on duration of songs
durations = df['duration_ms']
dis = sns.displot(durations, kde=True)
dis.figure.set_figwidth((20))
dis.figure.set_figheight((10))
plt.xlim(0, 0.8*10**6)
plt.title('Duration Distribution')
plt.xlabel('Duration')
plt.ylabel('Count')
plt.show()

#