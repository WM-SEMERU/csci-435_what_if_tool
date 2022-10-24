from transformers import AutoTokenizer, AutoModelForCausalLM


def run_pipeline(model: str, dataset: str, tokenizer: str) -> None:
    print("Pipeline initiated")
    selected_model = AutoModelForCausalLM.from_pretrained(model)
    selected_tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    selected_model.config.pad_token_id = selected_model.config.eos_token_id
    input_ids = selected_tokenizer(dataset, return_tensors="pt").input_ids

    outputs = selected_model.generate(
        input_ids, do_sample=False, max_length=50)

    output_strs = selected_tokenizer.batch_decode(
        outputs, skip_special_tokens=True)
    output_tkns = selected_tokenizer.tokenize(output_strs[0])

    return output_tkns
