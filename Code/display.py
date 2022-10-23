from jupyter_dash import JupyterDash

import dash
from dash import dcc
from dash import html
import pandas as pd


def display_bar_chart():

    
    app = JupyterDash(__name__)
    server = app.server

    app.layout = html.Div('Hello World')

    app.run_server(mode='inline')