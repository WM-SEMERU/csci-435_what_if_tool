from typing import List, TypedDict, Union
from uuid import uuid4
from jupyter_dash import JupyterDash
import plotly.express as px
from dash import dcc, html, Input, Output

from ..backend.model import preprocess
from ..backend.pipeline import Pipeline
from ..backend.pipeline_store import PipelineStore
from .layout import data_editor_components, graph_settings_components


class Dataset(TypedDict):
    id: int
    data: str


DUMMY_DATA = [{"label": str(uuid4()), "value": ["This is some chunk of code that I wish to analyze"]},
              {"label": str(uuid4()),
               "value": ["Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."]},
              {"label": str(uuid4()), "value": ["def foo(bar): print(bar) foo(123)"]}]

models = ["gpt2", "codeparrot/codeparrot-small",
          "codegen", "gpt-neo"]  # add codebert, neox?

pipes = PipelineStore()


def run_server(model: str, dataset: List[str], dataset_id: Union[str, None]) -> None:
    app = JupyterDash(__name__)

    # TODO: refactor for efficiency
    add_dataset = True
    for i in DUMMY_DATA:
        label, value = i["label"], i["value"]
        if value == dataset:
            dataset_id = label
            add_dataset = False

    if add_dataset:
        dataset_id = str(uuid4())
        DUMMY_DATA.append({"label": dataset_id, "value": dataset})

    # TODO: don't create new pipeline if identical one already exists!
    input_pipe = Pipeline(model, dataset, dataset_id)
    pipes.add_pipeline(input_pipe)
    pipes.run_pipelines()

    FLAT_DUMMY = [{"label": dummy_data["label"], "value": " ".join(
        dummy_data["value"])} for dummy_data in DUMMY_DATA]

    app.layout = html.Div([
        html.Div(data_editor_components, className="dataEditor"),
        html.Div(graph_settings_components(
            FLAT_DUMMY, " ".join(dataset), models, model), className="graphSettings"),
        html.Div([dcc.Graph(id="graph1"), dcc.Graph(
            id="graph2")], className="graph")
    ])

    # TODO: update so bar chart doesn't include input sequence in analyzed tokens! Only predicted tokens.
    # TODO: update so string representations of tokens are shown rather than tokens themselves
    @app.callback(Output("graph1", "figure"), Input("dataset_dropdown_1", "value"), Input("model_dropdown_1", "value"), Input("desc_stats_1", "value"))
    def update_bar_chart1(selected_dataset: Union[Dataset, str], selected_model: Union[str, None], selected_stat: str):
        selected_dataset_id = None
        for i in FLAT_DUMMY:
            label, value = i["label"], i["value"]
            if value == selected_dataset:
                selected_dataset_id = label

        # print(f'{selected_dataset_id} {selected_dataset} {selected_model}')
        df = preprocess(selected_model, selected_dataset,
                        selected_dataset_id, selected_stat)
        # print("\ndf: ", df)
        fig = px.bar(df, x="frequency", y="token", labels={
                     "frequency": str(selected_stat) + " token frequency"})
        return fig

    # @app.callback(Output("graph2", "figure"), Input("dataset_dropdown_2", "value"), Input("model_dropdown_2", "value"))
    # def update_bar_chart2(selected_dataset: Union[str, None], selected_model: Union[str, None]):
    #     selected_dataset_id = None
    #     for i in flattened_DUMMY:
    #         label, value = i["label"], i["value"]
    #         if value == selected_dataset:
    #             selected_dataset_id = label

    #     # print(f'{selected_dataset_id} {selected_dataset} {selected_model}')
    #     df = preprocess(tokenizer, selected_model,
    #                     selected_dataset, selected_dataset_id)
    #     # print("\ndf: ", df)
    #     fig = px.bar(df, x="frequency", y="token")
    #     return fig

    app.run_server(mode="inline", debug=True)
