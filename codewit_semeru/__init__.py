"""
codewit_semeru

What-if-tool Code. A Visual Tool for Understanding Machine Learning Models for Software Engineering
"""

__version__ = "0.0.4"
__author__ = 'WM-SEMERU'
__credits__ = 'College of William & Mary'


from typing import List
from .frontend.display import CodeWITServer


def WITCode(model: str = "gpt2", dataset: List[str] = [], dataset_id: str = None) -> None:
    server = CodeWITServer(model, dataset, dataset_id)
    server.run()
    # run_server(model, dataset, dataset_id)
