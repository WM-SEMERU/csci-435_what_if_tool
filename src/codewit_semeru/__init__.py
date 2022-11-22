"""
codewit_semeru

What-if-tool Code. A Visual Tool for Understanding Machine Learning Models for Software Engineering
"""

__version__ = "0.0.4"
__author__ = 'WM-SEMERU'
__credits__ = 'College of William & Mary'


from .frontend.display import run_server


def WITCode(tokenizer: str = "gpt2", model: str = "gpt2", dataset: str = "", dataset_id: str = None) -> None:
    run_server(tokenizer, model, dataset, dataset_id)
