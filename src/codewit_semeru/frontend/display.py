from typing import List, TypedDict, Union
from uuid import uuid4
from jupyter_dash import JupyterDash
import plotly.express as px
from dash import dcc, html, Input, Output
from bertviz import head_view

from ..backend.model import preprocess, get_bertviz
from ..backend.pipeline import Pipeline
from ..backend.pipeline_store import PipelineStore
from .layout import data_editor_components, graph_settings_components


DUMMY_DATA = [{"label": str(uuid4()), "value": ["This is some chunk of code that I wish to analyze"]},
              {"label": str(uuid4()),
               "value": ["Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."]},
              {"label": str(uuid4()), "value": ["def foo(bar): print(bar) foo(123)"]}]

models = ["gpt2", "codeparrot/codeparrot-small",
          "codegen", "gpt-neo"]  # add codebert, neox?

pipes = PipelineStore()

# TODO: convert to class to reduce redundancy of parameters


def update_data_and_chart(tokenizer: str, selected_model: Union[str, None], selected_dataset: List[str], selected_stat: str):
    selected_dataset_id = None
    for i in DUMMY_DATA:
        label, value = i["label"], i["value"]
        if value[0] == selected_dataset[0]:
            selected_dataset_id = label

    print(
        f"Processing {Pipeline.pipe_id(tokenizer, selected_model, selected_dataset_id)}\nPlease wait...")
    df = preprocess(tokenizer, selected_model,
                    selected_dataset, selected_dataset_id, selected_stat)
    # print("\ndf: ", df)
    fig = px.bar(df, x="frequency", y="token")
    return fig


def run_server(tokenizer: str, model: str, dataset: List[str], dataset_id: Union[str, None]) -> None:
    app = JupyterDash(__name__)

    # TODO: refactor for efficiency
    add_dataset = True
    for i in DUMMY_DATA:
        label, value = i["label"], i["value"]
        if value[0] == dataset[0]:
            dataset_id = label
            add_dataset = False

    if add_dataset:
        dataset_id = str(uuid4())
        DUMMY_DATA.append({"label": dataset_id, "value": dataset})

    # TODO: don't create new pipeline if identical one already exists!
    input_pipe = Pipeline(tokenizer, model, dataset, dataset_id)
    pipes.add_pipeline(input_pipe)
    pipes.run_pipelines()

    # run_pipeline(model, dataset, tokenizer)
    # html_head_view = get_bertviz()
    # with open("codewit_semeru/codewit_semeru/frontend/assets/head_view.html", 'w') as file:
    #     file.write(html_head_view.data)
    # bertviz_html = parse_head_view()

    app.layout = html.Div([
        html.Div(data_editor_components, className="dataEditor"),
        html.Div(graph_settings_components(
            DUMMY_DATA, dataset, models, model), className="graphSettings"),
        html.Div([dcc.Graph(id="graph1"), dcc.Graph(
            id="graph2")], className="graph"),
        # Attempt to add radio items to select some bertviz view
        # html.Div(dcc.RadioItems(["head", "neuron", "model"], id="bert_select")),
        # html.Div([get_bertviz()], className="bertviz"),
    ])
    # head_view(dataset, dataset)

    # TODO: update so bar chart doesn't include input sequence in analyzed tokens! Only predicted tokens.
    # TODO: update so string representations of tokens are shown rather than tokens themselves
    @app.callback(Output("graph1", "figure"), Input("dataset_dropdown_1", "value"), Input("model_dropdown_1", "value"), Input("desc_stats_1", "value"))
    def update_bar_graph1(selected_dataset: List[str], selected_model: Union[str, None], selected_stat: Union[str, None]):
        return update_data_and_chart(tokenizer, selected_model if selected_model else model, selected_dataset, selected_stat)

    """ @app.callback(Output("graph2", "figure"), Input("dataset_dropdown_2", "value"), Input("model_dropdown_2", "value"), Input("desc_stats_2", "value"))
    def update_bar_graph2(selected_dataset: Union[str, None], selected_model: Union[str, None], selected_stat: Union[str, None]):
        return update_data_and_chart(tokenizer, selected_model if selected_model else model, selected_dataset, selected_stat) """

    # @app.callback(Output("bertviz", "children"), Input("dataset_dropdown", "value"))
    # def update_bertviz(value):
    #     attention, input_tkns = get_bertviz()
    #     html_rep = head_view(attention, input_tkns, html_action='return')
    #     return html_rep

    # update_bar_chart(dataset if dataset else DUMMY_DATA[0])
    # update_bertviz(1)
    app.run_server(mode="inline", debug=True)
