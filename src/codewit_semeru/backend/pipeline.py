from typing import Union
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class Pipeline:
    # to-do https://github.com/tensorflow/tensorflow/issues/53529
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __init__(self, model: str, dataset: Union[str, int]) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, output_attentions=True).to(self.device)
        self.dataset = dataset
        self.output = None  
        self.attention = None
        self.output_strs = None
        self.output_tkns = None
        self.completed = False

        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.input_ids = self.tokenizer(
            dataset, return_tensors="pt").input_ids.to(self.device)
        self.input_tkns = self.tokenizer.convert_ids_to_tokens(
            self.input_ids[0])

    def run(self) -> None:
        # Weird interaction here where specifiying transformers generate pipeline + getting attention does not quite work...
        # to-do : figure out how to extract all necessary info from one pipeline run
        self.output = self.model.generate(
            self.input_ids, do_sample=False, max_length=50)
        self.test_output = self.model(self.input_ids)

        self.attention = self.test_output[-1]
        self.output_strs = self.tokenizer.batch_decode(
            self.output, skip_special_tokens=True)
        self.output_tkns = self.tokenizer.tokenize(self.output_strs[0])

        self.completed = True 
        # print(self.attention)
        print(self.input_tkns)
        print("Completed")
