from dash import dcc, html

filler = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer dictum hendrerit quam ac convallis. Maecenas laoreet nibh rutrum tortor porta, sed faucibus ex tincidunt. Sed sed est dolor. Fusce convallis dui sed tortor posuere scelerisque et non nisl. Maecenas sollicitudin non nisl ut lobortis. Proin ultrices vel erat quis ultricies. Nunc accumsan purus nibh, eu luctus odio eleifend id. Etiam eget lectus sed erat tincidunt imperdiet. Donec rutrum mauris lacinia eros ultrices, rutrum interdum tortor pulvinar. Donec id libero ut dolor ultrices maximus. Vivamus dictum ultrices metus in pharetra. In in viverra est. Praesent velit eros, viverra a ultricies quis, pellentesque in ipsum. Suspendisse lacus justo, placerat eget dignissim quis, laoreet non nisi."

data_editor_components = [
    html.P(
        ["Datapoint Name"],
        className="dataLabel"
    ),
    html.Div([
        html.Label(["Raw Input:"], htmlFor="input"),
        html.Textarea([filler], id="input", name="input", rows="4", cols="18")],
        className="dataField"),
    html.Div([
        html.Label(["Vectorized input 1:"], htmlFor="data1"),
        dcc.Input(type="text", id="data1", name="data1"),
        html.Label(["Vectorized input 2:"], htmlFor="data2"),
        dcc.Input(type="text", id="data2", name="data2"),
        html.Label(["Vectorized input 3:"], htmlFor="data3"),
        dcc.Input(type="text", id="data3", name="data3"),
        html.Label(["Vectorized input 4:"], htmlFor="data4"),
        dcc.Input(type="text", id="data4", name="data4"),
        html.Label(["Vectorized input 5:"], htmlFor="data5"),
        dcc.Input(type="text", id="data5", name="data5")
    ], className="dataField"),
    html.Div(["Predictions:", html.Br(), "[Insert table here]"], className="dataOutput")]

def graph_settings_components (datasets, dataset, models, model):
    return [
        html.Div([
            html.Div([
                "View:",
                dcc.Dropdown(['single graph', 'two graph comparison'], id="view_dropdown")]),
            html.Div([
                html.Div([
                    "Dataset:",
                    dcc.Dropdown(datasets, value = dataset, id="dataset_dropdown_1"),
                    "Model:",
                    dcc.Dropdown(models, id = "model_dropdown1", value = model),
                    "Descriptive Stat:",
                    dcc.Dropdown(["mean", "median",  "std dev", "mode", "max", "min"], id="desc_stats_1", value = "mean"),
                    "Num Tokens:",
                    dcc.Input(type="text", id="num_token_1"),
                    html.Br(),
                    "Num Input Sequences:",
                    dcc.Input(type="text", id="input_seq_1"),
                    html.Br(),
                    "Graph type:",
                    dcc.Dropdown(["code concept histogram", "basic token histogram", "bertviz"], id="graph_type_1")
                    ],
                    className="graphSettingsFill"),
                html.Div([
                    "Dataset:",
                    dcc.Dropdown(['Option 1', 'Option 2', 'Option 3'], id="dataset_dropdown_2"),
                    "Model:",
                    dcc.Dropdown(models, id="model_dropdown2"),
                    "Num Tokens:",
                    dcc.Input(type="text", id="num_token_2"),
                    html.Br(),
                    "Num Input Sequences:",
                    dcc.Input(type="text", id="input_seq_2"),
                    html.Br(),
                    "Graph type:",
                    dcc.Dropdown(["code concept histogram", "basic token histogram", "bertviz"], id="graph_type_2"),
                    "Descriptive Stats:",
                    dcc.Dropdown(["median", "mode", "max", "min", "std dev"], id="desc_stats_2"),
                    ],
                    className="graphSettingsFill")])
            ], className="graphSettings")
        ]

graph_display_components = [
    html.Div([
        html.Div([
            dcc.Graph(id="graph1")],
            className="graphFill"),
        html.Div([
            dcc.Graph(id="graph2")],
            className="graphFill")],
        className="graph")]
            

"""
graph_settings_components = lambda models: [
    "Dataset:",
    dcc.Dropdown(['Option 1', 'Option 2', 'Option 3'], id="dataset_dropdown"),
    "Model:",
    dcc.Dropdown(models),
]
"""
