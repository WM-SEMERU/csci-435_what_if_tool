from typing import List
from collections import Counter
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from .pipeline import Pipeline
from .pipeline_store import PipelineStore


def run_pipeline(model: str, dataset: str, tokenizer: str) -> None:
    print("Pipeline initiated")
    selected_model = AutoModelForCausalLM.from_pretrained(
        model, output_attentions=True)
    selected_tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    selected_model.config.pad_token_id = selected_model.config.eos_token_id
    input_ids = selected_tokenizer(dataset, return_tensors="pt").input_ids

    outputs = selected_model.generate(
        input_ids, do_sample=False, max_length=50)
    attention = outputs[-1]

    input_tkns = selected_tokenizer.convert_ids_to_tokens(input_ids[0])
    output_strs = selected_tokenizer.batch_decode(
        outputs, skip_special_tokens=True)
    output_tkns = selected_tokenizer.tokenize(output_strs[0])

    return output_tkns


def preprocess(model: str, dataset: str, tokenizer: str) -> List[str]:
    curr_pipe = Pipeline(tokenizer, model, dataset)
    curr_pipe.start()
    # output_tkns = run_pipeline(model, dataset, tokenizer)

    output_tkns = curr_pipe.output_tkns

    counts = Counter(output_tkns)
    token_freq = pd.DataFrame(
        counts.items(), columns=["token", "frequency"]
    ).sort_values(by="frequency", ascending=False)

    return token_freq.head(20)
