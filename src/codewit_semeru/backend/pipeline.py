from typing import List, Union
from uuid import uuid4
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class Pipeline:
    # to-do https://github.com/tensorflow/tensorflow/issues/53529
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def pipe_id(tokenizer: str, model: str, dataset_id: str) -> str:
        if dataset_id == None:
            dataset_id = str(uuid4())
        return "<>".join([tokenizer, model, dataset_id])

    def __init__(self, tokenizer: str, model: str, dataset: Union[str, int], dataset_id: str = None) -> None:
        if dataset_id == None:
            dataset_id = str(uuid4())
        self.id: str = Pipeline.pipe_id(tokenizer, model, dataset_id)
        print(self.id)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, output_attentions=True).to(self.device)
        self.dataset = dataset
        self.dataset_id = dataset_id

        self.output = None
        self.attention = None
        self.output_strs: List[str] = None
        self.output_tkns: List[str] = None

        self.completed: bool = False

        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.input_ids = self.tokenizer(
            dataset, return_tensors="pt").input_ids.to(self.device)
        self.input_tkns: List[str] = self.tokenizer.convert_ids_to_tokens(
            self.input_ids[0])

    def run(self) -> None:
        # Weird interaction here where specifiying transformers generate pipeline + getting attention does not quite work...
        # to-do : figure out how to extract all necessary info from one pipeline run
        self.output = self.model.generate(
            self.input_ids, do_sample=False, max_length=500)
        self.test_output = self.model(self.input_ids)

        self.attention = self.test_output[-1]
        self.output_strs = self.tokenizer.batch_decode(
            self.output, skip_special_tokens=True)
        self.output_tkns = self.tokenizer.tokenize(self.output_strs[0])

        self.completed = True
        # print(self.attention)
        print(f"Pipeline completed for pipe {self.id}")
