from copy import copy
from typing import List
import pandas as pd
import statistics
from .pipeline import Pipeline
from .pipeline_store import PipelineStore
from bertviz import head_view

pipes = PipelineStore()


""" def run_pipeline(tokenizer: str, model: str, dataset: str) -> None:
    print("Pipeline initiated")
    pipes.add_pipeline(Pipeline(model, dataset))
    print('Running...')
    pipes.run_pipelines() """


def preprocess(tokenizer: str, model: str, dataset: str, dataset_id: str, stat: str = "mean") -> List[str]:
    
    # print("model: ", model)
    # print("dataset: ", dataset)
    # print("dataset_id: ", dataset_id)
    pipe_id = Pipeline.pipe_id(tokenizer, model, dataset_id)
    pipe = pipes.get_pipeline(pipe_id)

    print("pipeid: ", pipe_id)

    if not pipe:
        pipe = Pipeline(tokenizer, model, dataset, dataset_id)
        pipes.add_pipeline(pipe)
        pipes.run_pipe(pipe.id)

    if stat == "mean":
        stats_func = statistics.mean
    elif stat == "median":
        stats_func = statistics.median
    elif stat == "std dev":
        stats_func = statistics.stdev
    elif stat == "max":
        stats_func = max
    elif stat == "min":
        stats_func = min
    elif stat == "mode":
        stats_func = statistics.mode
    else:
        raise ValueError("Supported statistics are mean, median, std dev, mode, max, and min. Please use one of them.")

    # output_tkns = pipe.output_tkns
    # print("pipe.output_tok_freqs: ", pipe.output_tok_freqs)
    output_tkn_freqs = copy(pipe.output_tok_freqs)
    for tkn in output_tkn_freqs:
        output_tkn_freqs[tkn] = stats_func(output_tkn_freqs[tkn])

    token_freq = pd.DataFrame(
        output_tkn_freqs.items(), columns=["token", "frequency"]
    ).sort_values(by="frequency", ascending=False)
    print(f"token_freq for {stat}:\n{token_freq}")

    return token_freq.head(20)

    #["median", "mean", "max", "min", "std dev"]

def get_bertviz():
    #Temporarily coded so that bertviz analyzes only the FIRST input sequence from the "dataset" WITCode() list parameter 
    attention, input_tkns = pipes.get_pipeline(
        0).attention[0], pipes.get_pipeline(0).input_tkns[0]
    return head_view(attention, input_tkns)
