from typing import List
from uuid import uuid4
from jupyter_dash import JupyterDash
from matplotlib.figure import Figure
import plotly.express as px
from dash import dcc, html, Input, Output

from ..backend.model import preprocess
from ..backend.pipeline import Pipeline
from ..backend.pipeline_store import PipelineStore
from .layout import data_editor_components, graph_settings_components


DUMMY_DATA = [{"label": str(uuid4()), "value": ["This is some chunk of code that I wish to analyze"]},
              {"label": str(uuid4()),
               "value": ["Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."]},
              {"label": str(uuid4()), "value": ["def foo(bar): print(bar) foo(123)"]}]

models = ["gpt2", "codeparrot/codeparrot-small",
          "Salesforce/codegen-350M-mono", "EleutherAI/gpt-neo-125M"]  # add codebert, neox?

pipes = PipelineStore()


class CodeWITServer():
    def __init__(self, model: str, dataset: List[str], dataset_id: str):
        self.app = JupyterDash(__name__)

        self.model_1 = model
        self.dataset_1 = dataset if dataset else DUMMY_DATA[0]["value"]
        
        self.model_2, self.dataset_2 = "", []

        dataset_id = next((d["label"] for d in DUMMY_DATA if d["value"][0] == dataset[0]), "")
        if not dataset_id:
            dataset_id = str(uuid4())
            DUMMY_DATA.append({"label": dataset_id, "value": dataset})

        input_pipe = Pipeline(model, dataset, dataset_id)
        pipes.add_pipeline(input_pipe)
        pipes.run_pipelines()

        self.FLAT_DUMMY = [{"label": dataset["label"], "value": " ".join(
            dataset["value"])} for dataset in DUMMY_DATA]

        self.app.layout = html.Div([
            #html.Div(data_editor_components, className="dataEditor"),
            html.Div(graph_settings_components(
                self.FLAT_DUMMY, " ".join(dataset), models, model), className="graphSettings"),
            html.Div([dcc.Graph(id="graph1"), dcc.Graph(
                id="graph2")], className="graph")
        ])

    def update_data_and_chart(self, selected_model: str, selected_dataset: str, selected_stat: str) -> Figure:
        # ? #60 - can this be done another way given dataset dropdown vs. input?
        dataset_id = next((d["label"] for d in self.FLAT_DUMMY if d["value"] == selected_dataset), "")
        selected_dataset_id = dataset_id if dataset_id else ""

        dataset = next((d["value"] for d in DUMMY_DATA if d["label"] == selected_dataset_id), [])
        selected_dataset = dataset if dataset else []

        if not selected_dataset:
            raise LookupError

        print(
            f"Processing {Pipeline.pipe_id(selected_model, selected_dataset_id)}\nPlease wait...")

        df = preprocess(selected_model, selected_dataset,
                        selected_dataset_id, selected_stat)
        print("Done!")
        
        fig = px.bar(df, x="frequency", y="token")
        return fig

    def run(self) -> None:
        # TODO: update so bar chart doesn't include input sequence in analyzed tokens! Only predicted tokens.
        # TODO: update so string representations of tokens are shown rather than tokens themselves
        @self.app.callback(Output("graph1", "figure"), Input("dataset_dropdown_1", "value"), Input("model_dropdown_1", "value"), Input("desc_stats_1", "value"))
        def update_bar_graph1(selected_dataset: List[str] = self.dataset_1, selected_model: str = self.model_1, selected_stat: str = "mean"):
            try:
                return self.update_data_and_chart(selected_model, selected_dataset, selected_stat)
            except LookupError:
                print("error: dataset not found!")
                return px.bar()

        @self.app.callback(Output("graph2", "figure"), Input("dataset_dropdown_2", "value"), Input("model_dropdown_2", "value"), Input("desc_stats_2", "value"))
        def update_bar_graph2(selected_dataset: List[str] = self.dataset_2, selected_model: str = self.model_2, selected_stat: str = "mean"):
            if selected_dataset and selected_model:
                try:
                    return self.update_data_and_chart(selected_model, selected_dataset, selected_stat)
                except LookupError:
                    print("error: dataset not found!")
            return px.bar()

        self.app.run_server(mode="inline", debug=True)

