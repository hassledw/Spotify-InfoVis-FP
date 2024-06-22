from dash import Dash, dcc, html, Input, Output, exceptions
import plotly.express as px
import pandas as pd
from spotify_utils import ConnectSpotifyItem, Song
from waveform_vis import plot_song_waveform

###################################
#    Data Creation and Selection  #
###################################
click_counter = 0
df = pd.read_csv("./spotify.csv").dropna()
df = df.drop(columns=["Unnamed: 0"])
numeric_df = df[["popularity", "duration_ms", "danceability", "energy", "loudness",
                   "speechiness", "acousticness", "instrumentalness", "liveness",
                   "valence", "tempo"]]
categories = df['track_genre'].unique().tolist()
genre_artists = df[df['track_genre'] == categories[0]]["artists"].unique().tolist()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash("Spotify Visualization", external_stylesheets=external_stylesheets, title="Spotify Visualization")

# pie data

all_pie_df = pd.DataFrame()

for genre in df["track_genre"].unique().tolist():
    genre_pie_df = df[df["track_genre"] == genre].value_counts('key', sort=True)
    genre_pie_df = pd.DataFrame(genre_pie_df).rename(columns={"count" : genre})
    all_pie_df = pd.concat([all_pie_df, genre_pie_df], axis=1).sort_index()

###########################
#     Spotify API init    #
###########################
spotify_creds = ConnectSpotifyItem()
song = Song(spotify_creds.token, songname="My heart will go on")

########################
#       App Layout     #
########################
scatter_x_input = dcc.Slider(
        id="feature-x",
        vertical=False,
        min=0,
        max=len(df.columns),
        value=8,
        step=1,
        marks={i : column for i, column in enumerate(df.columns)}
    )

scatter_y_input =  dcc.Slider(
        id="feature-y",
        vertical=True,
        min=0,
        max=len(df.columns),
        value=5,
        step=1,
        marks={i : column for i, column in enumerate(df.columns)}
    )

scatter_graph = dcc.Graph(id="scatter-graph")

genre_slider = dcc.Slider(
        id="genre-slider",
        min=0,
        max=len(df["track_genre"].unique()),
        step=1,
        value=0,
        marks={i: genre_name for i, genre_name in enumerate(categories) if i % 10 == 0},
        tooltip={
            "always_visible": False,
            "template": "{value}"
        }
    )

artist_range_slider = dcc.RangeSlider(
    id="artist-range-slider",
    min=0,
    max=len(genre_artists),
    step=1,
    value=[0, len(genre_artists)],
    marks={i : artist for i, artist in enumerate(genre_artists) if i % 100 == 0},
    tooltip={
        "always_visible": False,
        "template": "{value}"
    }
)

pi_graph = dcc.Graph(id="pi-graph")

pi_genre_dropdown = dcc.Dropdown(all_pie_df.columns,
                 id='genre_dropdown',
                 value=all_pie_df.columns[0],
                 multi=False,
                 clearable=False)

area_graph = dcc.Graph(id="area-graph")

area_checklist = dcc.Checklist(
    id="area-checklist",
    options=numeric_df.columns.unique().tolist(),
    value=numeric_df.columns.unique().tolist()[:3],
    inline=True
)

song_text_area = dcc.Textarea(
    id="song-text-area",
    placeholder="Enter Song Name",
    value="Simple Man",
    className="text-center",
    style={"height": "15%", "width": "100%"}
)

song_loading = dcc.Loading(
    id="loading-1",
    type="default",
    children=html.Div(id="loading-output-1")
)

song_box = html.Div([
                        html.Img(id="album-image", src="", height="15%", width="15%", style={"padding-left": "5%"}),
                        html.Plaintext(id="song-info", children=[""])
                    ], style={"display": "flex", "align-items": "center", "width": "45%", "border": "solid", "border-radius": "2%"})

waveform_graph = dcc.Graph(id="waveform-plot")

download_button = html.Button("Download Song Data (CSV)", id="song-csv-button")
download_feature = dcc.Download(id="download-song-csv")

app.layout = html.Div(children=[
    # graph 1: scatter
    dcc.Tabs([
        dcc.Tab(label="Home", children=[
            html.Div([
                html.Br(),
                html.H3(children='Spotify Dataset Scatter Visualization'),
                html.Br(),
                html.H6("Select a Desired Genre"),
                genre_slider,
                html.H6("Select Desired Artists"),
                artist_range_slider,
                html.Div([
                    scatter_y_input
                ], style={"width": "15%", "display": "inline-block", "position": "relative"}),
                html.Div([
                    scatter_graph
                ], style={"width": "85%", "display": "inline-block", "position": "relative"}),
                html.Div([
                    scatter_x_input
                ], style={"width": "85%", "float": "right"})
            ]),

            html.Br(),
            # graph 2: pie
            html.Div([
                html.Br(),
                html.H3(children='Spotify Dataset Pie Visualization'),
                html.Div([
                    html.P('Select Genre:'),
                    pi_genre_dropdown,
                    pi_graph,
                ], style={"width": "50%", "display": "inline-block", "position": "relative"}),

                html.Div([
                    html.H6("Select Feature"),
                    area_graph,
                    area_checklist
                ])
            ]),
        ]),

        dcc.Tab(label="Song Analysis", children=[
            html.Div([
                html.Br(),
                html.H1(children='Song Visualization'),
                html.Div([
                    html.Div([
                        html.Header(children=[html.H5('Type out a Song Name')]),
                        song_text_area,
                        song_loading
                    ], style={"display": "block", "width": "40%"}),
                    song_box
                ], style={"display": "flex", "gap": "10%", "align-items": "center"}),

                html.Div([
                    waveform_graph
                ], style={"width": "100%", "display": "inline-block", "position": "relative"}),

                html.Div([
                    download_button,
                    download_feature
                ], style={"text-align": "center"})
            ])
        ])
    ])
])

########################
#       Callbacks      #
########################
@app.callback(
    Output("scatter-graph", "figure"),
    Input("feature-x", "value"),
    Input("feature-y", "value"),
    Input("genre-slider", "value"),
    Input("artist-range-slider", "value")
)
def modify_scatter_graph(feature_x: int, feature_y: int, genre: int, artist_range: list):
    genre_df = df[df['track_genre'] == categories[genre]]
    genre_artists = genre_df["artists"].unique().tolist()
    artists = [genre_artists[i] for i in range(artist_range[0], artist_range[1])]

    fig = px.scatter(
        genre_df[genre_df["artists"].isin(artists)],
        x=df.columns[feature_x],
        y=df.columns[feature_y],
        hover_data={
            'artists': True,
            'track_name': True
        },
        color="artists",
        size_max=20,
        title=f"{df.columns[feature_x]} vs {df.columns[feature_y]} on {categories[genre]} genre"
    )

    fig.update_layout(
        title_font_size=30,
        title_font_color="blue",
        xaxis_title_font_color="green",
        yaxis_title_font_color="green"
    )

    fig.update_traces(marker=dict(size=10))

    return fig

@app.callback(
Output("artist-range-slider", "marks"),
    Output("artist-range-slider", "value"),
    Output("artist-range-slider", "max"),
    [Input("genre-slider", "value")]
)
def modify_artist_slider_labels(genre: int):
    genre_artists = df[df['track_genre'] == categories[genre]]["artists"].unique().tolist()
    marks = {i : artist for i, artist in enumerate(genre_artists) if i % 30 == 0}
    value = [0, len(genre_artists)]
    max = len(genre_artists)

    return marks, value, max


@app.callback(
    Output("pi-graph", "figure"),
    [Input("genre_dropdown", "value")]
)
def modify_chart(genre : str):
    p_labels = ['Key C', 'Key C#/Db', 'Key D', 'Key D#/Eb', 'Key E', 'Key F', 'Key F#/Gb', 'Key G', 'Key G#/Ab',
                'Key A', 'Key A#/Bb', 'Key B']

    pie_genre = all_pie_df[genre]
    colors = ["#000000","#0B3301","#166601","#218801","#29A311",
              "#32BF01","#3ADA04","#43F205","#4CFF0C","#66FF34",
              "#80FF53","#99FF71"]

    piechart = px.pie(
        title=f"Key Breakdown of {genre} genre",
        data_frame=pie_genre,
        values=pie_genre,
        names=p_labels,
    )

    piechart.update_layout(
        title_font_size=30,
        title_font_color="blue",
    )

    piechart.update_traces(
        marker=dict(colors=colors)
    )

    return piechart

@app.callback(
    [Output("waveform-plot", "figure"),
    Output("album-image", "src"),
    Output("song-info", "children"),
    Output("loading-output-1", "children")],
    [Input("song-text-area", "value")]
)
def plot_waveform(songname : str):
    spotify_creds = ConnectSpotifyItem()
    song = Song(spotify_creds.token, songname=songname)
    # song.display_song_data()
    audio_data = song.get_audio_analysis()
    fig = plot_song_waveform(song, audio_data)
    image = song.album_image_link
    songinfo = song.display_song_data()

    return fig, image, songinfo, ""
@app.callback(
    Output("download-song-csv", "data"),
    [Input("song-text-area", "value"),
     Input("song-csv-button", "n_clicks")],
    prevent_initial_call=True,
)
def download_csv(songname : str, click : int):
    global click_counter

    if click is None:
        raise exceptions.PreventUpdate

    elif click <= click_counter:
        raise exceptions.PreventUpdate

    else:
        spotify_creds = ConnectSpotifyItem()
        song = Song(spotify_creds.token, songname=songname)
        song_data_dict = song.create_dataset_entry()
        song_df = pd.DataFrame(song_data_dict, index=[0])
        click_counter = click

        print(click, click_counter)

    return dcc.send_data_frame(song_df.to_csv, f"{songname}.csv")

# @app.callback(
#     Output("area-graph", "figure"),
#     [Input("area-checklist", "value")]
# )
# def update_area_plot(radioval : str):
#     # areadf = df.copy()
#     # tempo_speed_categories = [0, 90, 140, 160, 200]
#     # tempo_speed_names = ["Slow", "Relaxed", "Medium", "Fast", "Extra Fast"]
#     # name_count = 0
#     #
#     # # init all TempoCategory values to None.
#     # areadf.loc[:, 'TempoCategory'] = "None"
#     #
#     # for i in range(0, len(tempo_speed_categories)):
#     #     begin_tempo = tempo_speed_categories[i]
#     #     end_tempo = None
#     #     tempo_cond = None
#     #
#     #     if i == len(tempo_speed_categories) - 1:
#     #         tempo_cond = (areadf["tempo"] >= begin_tempo)
#     #     else:
#     #         end_tempo = tempo_speed_categories[i + 1]
#     #         tempo_cond = (areadf["tempo"] >= begin_tempo) & (areadf["tempo"] <= end_tempo)
#     #
#     #     areadf.loc[tempo_cond, 'TempoCategory'] = tempo_speed_names[name_count]
#     #     name_count += 1
#     #
#     # areadf["TempoCategory"] = pd.Categorical(areadf["TempoCategory"], categories=tempo_speed_names, ordered=True)
#     # areadf = areadf.groupby("TempoCategory").agg(
#     #     {"valence": "mean", "energy": "mean", "danceability": "mean"}).sort_values(by="TempoCategory")
#     #
#     # # print(areadf.head())
#     #
#     # fig = plt.figure(figsize=(10, 10))
#     # ax = fig.add_subplot(1, 1, 1)
#     # areadf.plot(kind='area', alpha=0.5, color=["purple", "blue", "red"], ax=ax)
#     # ax.set_title("Average Valence, Energy, and Danceability by Tempo Categories")
#     # ax.set_xlabel("Tempo Category")
#     # ax.set_ylabel("Average Value")
#     # plt.show()
#     pass
if __name__ == "__main__":
    app.run_server(debug=True)