from typing import TypedDict, Union
from jupyter_dash import JupyterDash
import plotly.express as px
from dash import dcc, html, Input, Output
from bertviz import head_view
import dash_dangerously_set_inner_html

from ..backend.model import preprocess, get_bertviz, run_pipeline
from ..backend.pipeline_store import PipelineStore
from .layout import data_editor_components, graph_settings_components


class Dataset(TypedDict):
    id: int
    data: str


DUMMY_DATA = [{"label": 1, "value": "This is some chunk of code that I wish to analyze"},
              {"label": 2, "value": "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."},
              {"label": 3, "value": "def foo(bar): print(bar); foo(123)"}]


def run_server(model: str, dataset: Union[str, int], tokenizer: str) -> None:
    app = JupyterDash(__name__)
    pipe_store = PipelineStore()
    run_pipeline(model, dataset, tokenizer)
    # html_head_view = get_bertviz()
    # with open("codewit_semeru/codewit_semeru/frontend/assets/head_view.html", 'w') as file:
    #     file.write(html_head_view.data)
    # bertviz_html = parse_head_view()
    # server = app.server
    print('Done Pipeline Run')
    """ components = [
        dcc.Dropdown(
            id="dataset_dropdown",
            options=DUMMY_DATA,
            value=dataset,
            clearable=False,
        ),
        dcc.Graph(id="graph")] """
    # update_bar_chart(dataset if dataset else DUMMY_DATA[0])
    app.layout = html.Div([
        html.Div(data_editor_components, className="dataEditor"),
        html.Div(graph_settings_components, className="graphSettings"),
        html.Div([dcc.Graph(id="graph")], className="graph"),
        # Attempt to add radio items to select some bertviz view
        # html.Div(dcc.RadioItems(["head", "neuron", "model"], id="bert_select")),
        html.Div([get_bertviz()], className='bertviz'), 
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
    
    # @app.callback(Output("bertviz", "children"), Input("dataset_dropdown", "value"))
    # def update_bertviz(value):
    #     attention, input_tkns = get_bertviz()
    #     html_rep = head_view(attention, input_tkns, html_action='return')
    #     return html_rep
    
    update_bar_chart(dataset if dataset else DUMMY_DATA[0])
    # update_bertviz(1)
    app.run_server(mode="inline", debug=True)
