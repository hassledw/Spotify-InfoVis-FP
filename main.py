from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
'''
Modified from starter code: https://go.plotly.com/website
'''
###################################
#    Data Creation and Selection  #
###################################

df = pd.read_csv("./spotify.csv").dropna()
categories = df['track_genre'].unique().tolist()
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash("Spotify Visualization", external_stylesheets=external_stylesheets, title="Spotify Visualization")

# pie data
pie_genres = ["pop-film", "k-pop", "chill", "sad", "grunge", "indian", "anime", "emo", "sertanejo", "pop"]

all_pie_df = pd.DataFrame()

for genre in pie_genres:
    genre_pie_df = df[df["track_genre"] == genre].value_counts('key', sort=True)
    genre_pie_df = pd.DataFrame(genre_pie_df).rename(columns={"count" : genre})
    all_pie_df = pd.concat([all_pie_df, genre_pie_df], axis=1).sort_index()

########################
#       App Layout     #
########################

app.layout = html.Div(children=[
    # graph 1: scatter
    html.Div(
        [html.H1(children='Spotify Dataset Scatter Visualization'),
    dcc.RadioItems(
        id="feature-x",
        inline=True,
        options=df.columns,
        value=f"{df.columns[6]}",
    ),
    dcc.RadioItems(
        id="feature-y",
        inline=True,
        options=df.columns,
        value=f"{df.columns[8]}",
    ),
    dcc.Graph(id="scatter-graph"),
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
    html.Div([
        html.H1(children='Spotify Dataset Pie Visualization'),
        dcc.Graph(id="pi-graph"),
        html.P('Select Genre:'),
        dcc.Dropdown(all_pie_df.columns,
                 id='genre_dropdown',
                 value=all_pie_df.columns[0],
                 multi=False,
                 clearable=False),
    ]),
])

########################
#       Callbacks      #
########################
@app.callback(
    Output("scatter-graph", "figure"),
    Input("feature-x", "value"),
    Input("feature-y", "value"),
    Input("genre", "value"),
)
def modify_graph(feature_x: str, feature_y: str, genre: int):
    fig = px.scatter(
        df[df['track_genre'] == categories[genre]],
        x=feature_x,
        y=feature_y,
        text="track_name",
        size_max=20,
        title=f"{feature_x} vs {feature_y} on {categories[genre]} genre",
    )

    fig.update_traces(marker=dict(size=10))

    return fig

@app.callback(
    Output("pi-graph", "figure"),
    [Input("genre_dropdown", "value")]
)
def modify_chart(genre):
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


if __name__ == "__main__":
    app.run_server(debug=True)