import os

from IPython.core.display import display, HTML, Javascript
from bertviz import head_view

def start():
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    vis_html = open(os.path.join(__location__, 'app.html')).read()
    vis_js = open(os.path.join(__location__, 'app.js')).read()
    css = css_styling(__location__)
    display(HTML(vis_html))
    display(HTML('<style>{}</style>'.format(css)))

    display(Javascript(vis_js))

def css_styling(location):
    return open(os.path.join(location, 'custom.css')).read()
