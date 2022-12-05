import sys
import os
import json
import requests
from collections import Counter, defaultdict
from dotenv import load_dotenv
from typing import List
from uuid import uuid4
from transformers import AutoTokenizer
import torch

path = f"{sys.path[0]}/codewit_semeru/backend/config/.env"
load_dotenv(path)

HF_API_KEY = os.getenv("HF_API_TOKEN")
headers = {"Authorization": f"Bearer {HF_API_KEY}"}


class Pipeline:
    # to-do https://github.com/tensorflow/tensorflow/issues/53529
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def pipe_id(model: str, dataset_id: str) -> str:
        return "<>".join([model, dataset_id])

    def __init__(self, model: str, dataset: List[str], dataset_id: str = None) -> None:
        if dataset_id == None:
            dataset_id = str(uuid4())
        self.id: str = Pipeline.pipe_id(model, dataset_id)

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = model
        self.dataset: List[str] = dataset
        self.dataset_id = dataset_id

        self.api_url = f"https://api-inference.huggingface.co/models/{self.model}"

        self.output_tok_freqs = defaultdict(list)

        self.completed: bool = False

    def query_model(self, payload: str):
        print(payload)
        if self.model == "Salesforce/codegen-350M-mono":
            # for i in range(len(payload)):
            data = {"inputs": payload}
        else:
            data = json.dumps(payload)
        response = requests.request(
            "POST", self.api_url, headers=headers, data=data)
        return json.loads(response.content.decode("utf-8"))

    # TODO: Update so output doesn't contain input sequence!

    def run(self) -> None:
        # Weird interaction here where specifiying transformers generate pipeline + getting attention does not quite work...
        # to-do : figure out how to extract all necessary info from one pipeline run
        data = self.query_model(self.dataset)
        print(data)
        output_strs = list(map(lambda res: res[0]["generated_text"], data))
        output_tkns = list(map(self.tokenizer.tokenize, output_strs))

        for tkns in output_tkns:
            cts = Counter(tkns)
            for tkn in cts:
                self.output_tok_freqs[tkn].append(cts[tkn])

        # Add 0 freq counts for tokens which were not within all predicted sequences
        for tkn in self.output_tok_freqs:
            seq_diff = len(output_tkns) - len(self.output_tok_freqs[tkn])
            self.output_tok_freqs[tkn].extend([0] * seq_diff)                
        
        self.completed = True
        print(f"Pipeline completed for pipe {self.id}")
