from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
from spotify_utils import ConnectSpotifyItem, Song
from waveform_vis import plot_song_waveform
###################################
#    Data Creation and Selection  #
###################################

df = pd.read_csv("./spotify.csv").dropna()
df = df.drop(columns=["Unnamed: 0"])
categories = df['track_genre'].unique().tolist()
genre_artists = df[df['track_genre'] == categories[0]]["artists"].unique().tolist()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash("Spotify Visualization", external_stylesheets=external_stylesheets, title="Spotify Visualization")

# pie data
pie_genres = ["pop-film", "k-pop", "chill", "sad", "grunge", "indian", "anime", "emo", "sertanejo", "pop"]

all_pie_df = pd.DataFrame()

for genre in pie_genres:
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
        marks={i: genre_name for i, genre_name in enumerate(categories) if i % 10 == 0}
        # tooltip={
        #     "always_visible": True,
        #     "template": "categories[{value}]"
        # }
    )

artist_range_slider = dcc.RangeSlider(
    id="artist-range-slider",
    min=0,
    max=len(genre_artists),
    step=1,
    value=[0, len(genre_artists)],
    marks={i : artist for i, artist in enumerate(genre_artists) if i % 100 == 0}
)

pi_graph = dcc.Graph(id="pi-graph")

pi_genre_dropdown = dcc.Dropdown(all_pie_df.columns,
                 id='genre_dropdown',
                 value=all_pie_df.columns[0],
                 multi=False,
                 clearable=False)

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
                pi_graph,
                html.P('Select Genre:'),
                pi_genre_dropdown
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
                ], style={"width": "50%", "display": "inline-block", "position": "relative"})
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
        title=f"{df.columns[feature_x]} vs {df.columns[feature_y]} on {categories[genre]} genre",
    )

    fig.update_layout(
        title_font_size=30,
        title_font_color="red",
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
    #p_labels = ['Key C', 'Key C#/Db', 'Key D', 'Key D#/Eb', 'Key E', 'Key F', 'Key F#/Gb', 'Key G', 'Key G#/Ab',
                #'Key A', 'Key A#/Bb', 'Key B']

    pie_genre = all_pie_df[genre]
    # print(pie_genre)
    piechart = px.pie(
        data_frame=pie_genre,
        values=pie_genre,
        names=pie_genre.index
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
    song = Song(spotify_creds.token, songname=songname)
    song.display_song_data()
    audio_data = song.get_audio_analysis()
    fig = plot_song_waveform(song, audio_data)
    image = song.album_image_link
    songinfo = song.display_song_data()

    return fig, image, songinfo, ""

if __name__ == "__main__":
    app.run_server(debug=True)