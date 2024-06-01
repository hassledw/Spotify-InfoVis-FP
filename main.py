from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
'''
Modified from starter code: https://go.plotly.com/website
'''

df = pd.read_csv("./spotify.csv").dropna()
categories = df['track_genre'].unique().tolist()

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Spotify Dataset Visualization"),
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
    ]
)

@app.callback(
    Output("graph", "figure"),
    Input("pos-x", "value"),
    Input("pos-y", "value"),
    Input("genre", "value"),
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

if __name__ == "__main__":
    app.run_server(debug=True)