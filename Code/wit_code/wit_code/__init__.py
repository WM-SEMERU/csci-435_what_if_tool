"""
wit_code

What-if-tool Code. A Visual Tool for Understanding Machine Learning Models for Software Engineering
"""

__version__ = "0.1.0"
__author__ = 'Daniel Rodriguez Cardenas'
__credits__ = 'College of William & Mary'


from .display import run_server


def WITCode(model: str = "gpt2", dataset: str = "", tokenizer: str = "gpt2") -> None:
    run_server(model, dataset, tokenizer)
