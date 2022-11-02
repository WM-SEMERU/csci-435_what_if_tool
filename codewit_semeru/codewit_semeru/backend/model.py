from typing import List
from collections import Counter
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from .pipeline import Pipeline
from .pipeline_store import PipelineStore
from bertviz import head_view

pipes = PipelineStore()

def run_pipeline(model: str, dataset: str, tokenizer: str) -> None:
    print("Pipeline initiated")
    pipes.addPipeline(Pipeline(tokenizer, model, dataset))
    print('Running...')
    pipes.runPipelines()


def preprocess(model: str, dataset: str, tokenizer: str) -> List[str]:
    # pipes.addPipeline(Pipeline(tokenizer, model, dataset))
    # pipes.runPipelines()
    # curr_pipe = Pipeline(tokenizer, model, dataset)
    # curr_pipe.start()
    # output_tkns = run_pipeline(model, dataset, tokenizer)

    output_tkns = pipes.getPipeline(0).output_tkns
    # pipes.removePipeline(0)
    counts = Counter(output_tkns)
    token_freq = pd.DataFrame(
        counts.items(), columns=["token", "frequency"]
    ).sort_values(by="frequency", ascending=False)

    return token_freq.head(20)

def get_bertviz():
    attention, input_tkns = pipes.getPipeline(0).attention, pipes.getPipeline(0).input_tkns
    return head_view(attention, input_tkns)
