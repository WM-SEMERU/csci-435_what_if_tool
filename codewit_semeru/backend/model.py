from typing import List
import pandas as pd
import statistics
from .pipeline import Pipeline
from .pipeline_store import PipelineStore


pipes = PipelineStore()

def stats_func(stat: str):
    if stat == "mean":
        return statistics.mean
    elif stat == "median":
        return statistics.median
    elif stat == "std dev":
        return statistics.stdev
    elif stat == "max":
        return max
    elif stat == "min":
        return min
    elif stat == "mode":
        return statistics.mode
    else:
        raise ValueError

def preprocess(model: str, dataset: List[str], dataset_id: str, stat: str, graph: str) -> pd.DataFrame:
    pipe = pipes.get_pipeline(Pipeline.pipe_id(model, dataset_id))
    if not pipe:
        pipe = Pipeline(model, dataset, dataset_id)
        pipes.add_pipeline(pipe)
        pipes.run_pipe(pipe.id)

    token_freq = pd.DataFrame()
    output_tkn_freqs = {}
    if graph == "basic_token_hist":
        try:
            stat_func = stats_func(stat)
            output_tkn_freqs = {tkn: stat_func(freqs) for tkn, freqs in pipe.output_tok_freqs.items()}

            token_freq = pd.DataFrame.from_dict(output_tkn_freqs, orient="index", columns=[
                                                "frequency"]).rename_axis("token").reset_index()
            token_freq = token_freq.sort_values(by="frequency", ascending=False)
        except ValueError:
            print("Supported statistics are mean, median, std dev, mode, max, and min. Please use one of them.")
            return pd.DataFrame()

    else:
        output_tkn_freqs = pipe.output_tok_freqs
        token_freq = pd.DataFrame.from_dict(output_tkn_freqs, orient="index", columns=[
                                            f"output sequence {i}" for i in range(1, len(pipe.dataset) + 1)]).rename_axis("token").reset_index()    

    return token_freq.head(20)
