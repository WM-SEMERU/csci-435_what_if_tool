from typing import TypedDict, Union
from jupyter_dash import JupyterDash
import plotly.express as px
from dash import dcc, html, Input, Output
from bertviz import head_view

from ..backend.model import preprocess
from .layout import data_editor_components, graph_settings_components


class Dataset(TypedDict):
    id: int
    data: str


DUMMY_DATA = [{"label": 1, "value": "This is some chunk of code that I wish to analyze"},
              {"label": 2, "value": "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."},
              {"label": 3, "value": "def foo(bar): print(bar); foo(123)"}]


def run_server(model: str, dataset: Union[str, int], tokenizer: str) -> None:
    app = JupyterDash(__name__)
    # server = app.server

    """ components = [
        dcc.Dropdown(
            id="dataset_dropdown",
            options=DUMMY_DATA,
            value=dataset,
            clearable=False,
        ),
        dcc.Graph(id="graph")] """

    app.layout = html.Div([
        html.Div(data_editor_components, className="dataEditor"),
        html.Div(graph_settings_components, className="graphSettings"),
        html.Div([dcc.Graph(id="graph")], className="graph")
    ])
    # head_view(dataset, dataset)

    @app.callback(Output("graph", "figure"), Input("dataset_dropdown", "value"))
    def update_bar_chart(selected_dataset: Union[Dataset, str]):
        dataset = selected_dataset if isinstance(
            selected_dataset, str) else selected_dataset.data
        df = preprocess(model, dataset, tokenizer)
        print(df)
        fig = px.bar(df, x="frequency", y="token")
        return fig

    update_bar_chart(dataset if dataset else DUMMY_DATA[0])

    app.run_server(mode="inline", debug=True)
