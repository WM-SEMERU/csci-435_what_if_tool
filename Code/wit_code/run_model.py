# from unittest.util import _MAX_LENGTH
from transformers import AutoTokenizer, AutoModelForCausalLM

from .display import display_bar_chart


def run_pipeline(model, dataset, tokenizer):
    print("Pipeline initiated")
    selected_model = AutoModelForCausalLM.from_pretrained(model)
    selected_tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    selected_model.config.pad_token_id = selected_model.config.eos_token_id
    input_ids = selected_tokenizer(dataset, return_tensors="pt").input_ids

    outputs = selected_model.generate(
        input_ids, do_sample=False, max_length=50)

    answer = selected_tokenizer.batch_decode(outputs, skip_speical_tokens=True)

    answer_tokens = selected_tokenizer.tokenize(answer[0])

    # print(answer_tokens)
    display_bar_chart(answer_tokens)
