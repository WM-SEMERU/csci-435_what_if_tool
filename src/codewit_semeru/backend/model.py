from typing import List
from collections import Counter
import pandas as pd
from .pipeline import Pipeline
from .pipeline_store import PipelineStore
from bertviz import head_view

pipes = PipelineStore()


""" def run_pipeline(tokenizer: str, model: str, dataset: str) -> None:
    print("Pipeline initiated")
    pipes.add_pipeline(Pipeline(tokenizer, model, dataset))
    print('Running...')
    pipes.run_pipelines() """


def preprocess(tokenizer: str, model: str, dataset: str, dataset_id: str) -> List[str]:
    pipe_id = Pipeline.pipe_id(tokenizer, model, dataset_id)
    pipe = pipes.get_pipeline(pipe_id)

    if not pipe:
        pipe = Pipeline(tokenizer, model, dataset, dataset_id)
        pipes.add_pipeline(pipe)
        pipes.run_pipe(pipe_id)

    output_tkns = pipe.output_tkns

    counts = Counter(output_tkns[0]) #Temporarily coded to analyze only the FIRST input sequence from the "dataset" WITCode() list parameter
    token_freq = pd.DataFrame(
        counts.items(), columns=["token", "frequency"]
    ).sort_values(by="frequency", ascending=False)

    return token_freq.head(20)


def get_bertviz():
    #Temporarily coded so that bertviz analyzes only the FIRST input sequence from the "dataset" WITCode() list parameter 
    attention, input_tkns = pipes.get_pipeline(
        0).attention[0], pipes.get_pipeline(0).input_tkns[0]
    return head_view(attention, input_tkns)
