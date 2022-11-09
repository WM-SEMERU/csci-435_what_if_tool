from typing import List
from collections import Counter
import pandas as pd
from .pipeline import Pipeline
from .pipeline_store import PipelineStore
from bertviz import head_view

pipes = PipelineStore()


def run_pipeline(model: str, dataset: str, tokenizer: str) -> None:
    print("Pipeline initiated")
    pipes.add_pipeline(Pipeline(tokenizer, model, dataset))
    print('Running...')
    pipes.run_pipelines()


def preprocess() -> List[str]:
    output_tkns = pipes.get_pipeline(0).output_tkns

    counts = Counter(output_tkns)
    token_freq = pd.DataFrame(
        counts.items(), columns=["token", "frequency"]
    ).sort_values(by="frequency", ascending=False)

    return token_freq.head(20)


def get_bertviz():
    attention, input_tkns = pipes.get_pipeline(
        0).attention, pipes.get_pipeline(0).input_tkns
    return head_view(attention, input_tkns)
