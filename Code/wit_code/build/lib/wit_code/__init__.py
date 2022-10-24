"""
wit_code

What-if-tool Code. A Visual Tool for Understanding Machine Learning Models for Software Engineering
"""

__version__ = "0.1.0"
__author__ = 'Daniel Rodriguez Cardenas'
__credits__ = 'College of William & Mary'


from .run_model import run_pipeline


def WITCode(model, dataset, tokenizer):
    run_pipeline(model, dataset, tokenizer)
