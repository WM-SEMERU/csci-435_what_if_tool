from typing import List
import pandas as pd
import statistics
from .pipeline import Pipeline
from .pipeline_store import PipelineStore


pipes = PipelineStore()


def preprocess(model: str, dataset: str, dataset_id: str, stat: str = "mean") -> List[str]:
    pipe = pipes.get_pipeline(Pipeline.pipe_id(model, dataset_id))
    if not pipe:
        pipe = Pipeline(model, dataset, dataset_id)
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
        raise ValueError(
            "Supported statistics are mean, median, std dev, mode, max, and min. Please use one of them.")

    output_tkn_freqs = {tkn: stats_func(freqs)
                        for tkn, freqs in pipe.output_tok_freqs.items()}
    token_freq = pd.DataFrame.from_dict(output_tkn_freqs, orient="index", columns=[
                                        "frequency"]).rename_axis("token").reset_index()
    token_freq = token_freq.sort_values(by="frequency", ascending=False)
    print(f"token_freq for {stat}:\n{token_freq}")

    return token_freq.head(20)
