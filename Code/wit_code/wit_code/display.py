from collections import Counter
from typing import TypedDict, List, Union
import pandas as pd
from jupyter_dash import JupyterDash
import plotly.express as px
from dash import dcc, html, Input, Output

from .model import run_pipeline


class Dataset(TypedDict):
    id: int
    data: str


DUMMY_DATA = [{"label": 1, "value": "This is some chunk of code that I wish to analyze"},
              {"label": 2, "value": "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."},
              {"label": 3, "value": "def foo(bar): print(bar); foo(123)"}]


def run_server(model: str, dataset: Union[str, int], tokenizer: str) -> None:
    app = JupyterDash(__name__)
    # server = app.server

    components = [
        dcc.Dropdown(
            id="dataset_dropdown",
            options=DUMMY_DATA,
            value=dataset,
            clearable=False,
        ),
        dcc.Graph(id="graph")]

    app.layout = html.Div(components)

    # TODO: user interactivity with listener functions
    @app.callback(Output("graph", "figure"), Input("dataset_dropdown", "value"))
    def update_bar_chart(selected_dataset: Union[Dataset, str]):
        print("Testing")
        dataset = selected_dataset if isinstance(
            selected_dataset, str) else selected_dataset.data
        df = preprocess(model, dataset, tokenizer)
        fig = px.bar(df, x="frequency", y="token")
        return fig

    update_bar_chart(dataset if dataset else DUMMY_DATA[0])

    app.run_server(mode="inline", debug=True)


def preprocess(model: str, dataset: str, tokenizer: str) -> List[str]:
    output_tkns = run_pipeline(model, dataset, tokenizer)

    counts = Counter(output_tkns)
    token_freq = pd.DataFrame(
        counts.items(), columns=["token", "frequency"]
    ).sort_values(by="frequency", ascending=False)

    return token_freq.head(20)
