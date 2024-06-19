from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
'''
Modified from starter code: https://go.plotly.com/website
'''

df = pd.read_csv("./spotify.csv").dropna()
categories = df['track_genre'].unique().tolist()

app = Dash(__name__)

# pie data
alldf = df.value_counts('key').sort_index()
popfilmdf = df[df['track_genre'] == "pop-film"].value_counts('key').sort_index()
kpopdf = df[df['track_genre'] == "k-pop"].value_counts('key').sort_index()
chilldf = df[df['track_genre'] == "chill"].value_counts('key').sort_index()
saddf = df[df['track_genre'] == "sad"].value_counts('key').sort_index()
grungedf = df[df['track_genre'] == "grunge"].value_counts('key').sort_index()
indiandf = df[df['track_genre'] == "indian"].value_counts('key').sort_index()
animedf = df[df['track_genre'] == "anime"].value_counts('key').sort_index()
emodf = df[df['track_genre'] == "emo"].value_counts('key').sort_index()
sertanejodf = df[df['track_genre'] == "sertanejo"].value_counts('key').sort_index()
popdf = df[df['track_genre'] == "pop"].value_counts('key').sort_index()
new_genre_df = pd.concat([alldf, popfilmdf, kpopdf, chilldf, saddf, grungedf, indiandf, animedf, emodf, sertanejodf, popdf], axis=1,
                         keys=['all genres', 'pop-film', 'k-pop', 'chill', 'sad', 'grunge', 'indian', 'anime', 'emo', 'sertanejo', 'pop'])

app.layout = html.Div(children=[
    # graph 1: scatter
    html.Div(
        [html.H1(children='Spotify Dataset Scatter Visualization'),
    dcc.RadioItems(
        id="pos-x",
        inline=True,
        options=df.columns,
        value=f"{df.columns[6]}",
    ),
    dcc.RadioItems(
        id="pos-y",
        inline=True,
        options=df.columns,
        value=f"{df.columns[8]}",
    ),
    dcc.Graph(id="graph"),
    dcc.Slider(
        id="genre",
        min=0,
        max=len(df["track_genre"].unique()),
        step=1,
        value=0,
        marks={i: genre_name for i, genre_name in enumerate(categories)}
    ),
]),
    # graph 2: pie
    html.Div(
        [html.H1(children='Spotify Dataset Pie Visualization'),
    dcc.Graph(id="pi-graph"),
    html.P('Select Genre:'),
    dcc.Dropdown(new_genre_df.columns, id='genre_dropdown', multi=False, clearable=False),
    ]),
])

@app.callback(
    Output("graph", "figure"),
    Input("pos-x", "value"),
    Input("pos-y", "value"),
    Input("genre", "value"),
)
@app.callback(
    Output("pi-graph", "figure"),
    [Input("genre_dropdown", "value")]
)

def modify_graph(pos_x: str, pos_y: str, genre: int):
    fig = px.scatter(
        df[df['track_genre'] == categories[genre]],
        x=pos_x,
        y=pos_y,
        text="track_name",
        size_max=20,
        title=f"{pos_x} vs {pos_y} on {categories[genre]} genre",
    )

    fig.update_traces(marker=dict(size=10))

    return fig

def modify_chart(genre_dropdown):
    #p_labels = ['Key C', 'Key C#/Db', 'Key D', 'Key D#/Eb', 'Key E', 'Key F', 'Key F#/Gb', 'Key G', 'Key G#/Ab',
                #'Key A', 'Key A#/Bb', 'Key B']
    piechart = px.pie(data_frame=new_genre_df, names=genre_dropdown)

    return piechart

if __name__ == "__main__":
    app.run_server(debug=True)