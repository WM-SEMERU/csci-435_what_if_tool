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


def graph_settings_components(datasets, dataset, models, model):
    return [
        "Dataset:",
        dcc.Dropdown(datasets, id="dataset_dropdown", value=dataset[0]),
        "Model:",
        dcc.Dropdown(models, id="model_dropdown",
                     value=model if model in models else None),
    ]
