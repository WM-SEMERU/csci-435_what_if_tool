from typing import List
from collections import Counter
import pandas as pd
from .pipeline import Pipeline
from .pipeline_store import PipelineStore
from bertviz import head_view

pipes = PipelineStore()


def run_pipeline(model: str, dataset: List[str]) -> None:
    print("Pipeline initiated")
    pipes.add_pipeline(Pipeline(model, dataset))
    print('Running...')
    pipes.run_pipelines()


def preprocess() -> List[str]:
    '''
    Returns dataframe of frequencies
    '''
    output_tkns = pipes.get_pipeline(0).output_tkns

    counts = Counter(output_tkns[0])    #Temporarily coded to analyze only the FIRST input sequence from the "dataset" WITCode() list parameter 
    token_freq = pd.DataFrame(
        counts.items(), columns=["token", "frequency"]
    ).sort_values(by="frequency", ascending=False)

    return token_freq.head(20)


def get_bertviz():
    #Temporarily coded so that bertviz analyzes only the FIRST input sequence from the "dataset" WITCode() list parameter 
    attention, input_tkns = pipes.get_pipeline(
        0).attention[0], pipes.get_pipeline(0).input_tkns[0]
    return head_view(attention, input_tkns)
