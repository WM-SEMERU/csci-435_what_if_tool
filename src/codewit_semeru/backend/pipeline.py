from collections import Counter, defaultdict
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

    def __init__(self, tokenizer: str, model: str, dataset: List[str], dataset_id: str = None) -> None:
        if dataset_id == None:
            dataset_id = str(uuid4())
        self.id: str = Pipeline.pipe_id(tokenizer, model, dataset_id)
        print(self.id)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, output_attentions=True).to(self.device)
        self.dataset = dataset
        self.dataset_id = dataset_id

        self.output = []  
        self.test_output = []
        self.attention = []
        self.output_strs: List[str] = []
        self.output_tkns: List[str] = []
        self.output_tok_freqs = defaultdict(list)

        self.completed: bool = False

        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.input_ids = []
        self.input_tkns = []
        for i in range(len(dataset)):
            self.input_ids.append(self.tokenizer(
                dataset[i], return_tensors="pt").input_ids.to(self.device))
            self.input_tkns.append(self.tokenizer.convert_ids_to_tokens(
                self.input_ids[i][0]))

    #TODO: Update so output doesn't contain input sequence!
    def run(self) -> None:
        # Weird interaction here where specifiying transformers generate pipeline + getting attention does not quite work...
        # to-do : figure out how to extract all necessary info from one pipeline run
        for i in range(len(self.dataset)):
            self.output.append(self.model.generate(
                self.input_ids[i], do_sample=False, max_new_tokens=50))
            self.test_output.append(self.model(self.input_ids[i]))
            self.attention.append(self.test_output[i][-1])
            self.output_strs.append(self.tokenizer.batch_decode(
                self.output[i], skip_special_tokens=True))
            self.output_tkns.append(self.tokenizer.tokenize(self.output_strs[i][0]))

        for tokens in self.output_tkns:  
            counts = Counter(tokens)
            for token in counts:
                self.output_tok_freqs[token].append(counts[token])
        # print("output_tok_freqs1: ", self.output_tok_freqs)

        #Add 0 freq counts for tokens which were not within all predicted sequences
        for token in self.output_tok_freqs:
            for _ in range(len(self.output_tkns) - len(self.output_tok_freqs[token])):
                self.output_tok_freqs[token].append(0)
        # print("output_tok_freqs2: ", self.output_tok_freqs)

        self.completed = True
        # print("output_strs: ",self.output_strs)
        # print(self.attention)
        print(f"Pipeline completed for pipe {self.id}")
