from collections import Counter
import pandas as pd
from jupyter_dash import JupyterDash
import plotly.express as px
from dash import dcc, html, Input, Output


def display_bar_chart(dataset):

    app = JupyterDash(__name__)
    # server = app.server

    app.layout = html.Div(
        [
            dcc.Dropdown(
                id="dropdown",
                options=["Current"],
                value="Fri",
                clearable=False,
            ),
            dcc.Graph(id="graph"),
        ]
    )

    @app.callback(Output("graph", "figure"), Input("dropdown", "value"))
    def update_bar_chart(day):
        df = preprocess(dataset)
        fig = px.bar(df, x="frequency", y="token")
        return fig

    app.run_server(mode="inline", debug=True)


def preprocess(tokens):
    counts = Counter(tokens)
    token_freq = pd.DataFrame(
        counts.items(), columns=["token", "frequency"]
    ).sort_values(by="frequency", ascending=False)
    return token_freq.head(20)
