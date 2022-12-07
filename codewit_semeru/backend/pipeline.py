import sys
import os
import json
import requests
import time
from collections import Counter, defaultdict
from dotenv import load_dotenv
from typing import List
from transformers import AutoTokenizer
import torch

path = f"{sys.path[0]}/codewit_semeru/backend/config/.env"
load_dotenv(path)

HF_API_KEY = os.getenv("HF_API_TOKEN")
assert HF_API_KEY is not None

headers = {"Authorization": f"Bearer {HF_API_KEY}"}


class Pipeline:
    # to-do https://github.com/tensorflow/tensorflow/issues/53529
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def pipe_id(model: str, dataset_id: str) -> str:
        return "<>".join([model, dataset_id])

    def __init__(self, model: str, dataset: List[str], dataset_id: str = "") -> None:
        self.model: str = model
        self.dataset: List[str] = dataset
        self.dataset_id: str = dataset_id

        self.id: str = Pipeline.pipe_id(model, dataset_id)

        self.tokenizer = AutoTokenizer.from_pretrained(model)

        self.api_url = f"https://api-inference.huggingface.co/models/{self.model}"

        self.output_tok_freqs = defaultdict(list)

        self.completed: bool = False

    def query_model(self):
        data = json.dumps({"inputs": self.dataset})
        response = requests.request(
            "POST", self.api_url, headers=headers, data=data)
        return json.loads(response.content.decode("utf-8"))

    # TODO: Update so output doesn't contain input sequence!

    def run(self) -> None:
        res = self.query_model()
        while type(res) is dict and res["error"]:
            print("error: ", res["error"])
            if "estimated_time" not in res:
                raise RuntimeError("pipeline run")
            
            print("Retrying in ", res["estimated_time"], "seconds")
            time.sleep(res["estimated_time"])
            print("Retrying...")
            res = self.query_model()

        # print(res)
        output_seqs = [data[0]["generated_text"] for data in res]
        output_tkns = [self.tokenizer.tokenize(seq) for seq in output_seqs]

        for tkns in output_tkns:
            cts = Counter(tkns)
            for tkn in cts:
                self.output_tok_freqs[tkn].append(cts[tkn])
        # print("output_tok_freqs1: ", self.output_tok_freqs)


        # add 0 freq counts for tokens which were not within all predicted sequences
        for tkn in self.output_tok_freqs:
            seq_diff = len(output_tkns) - len(self.output_tok_freqs[tkn])
            self.output_tok_freqs[tkn].extend([0] * seq_diff)                
        # print("output_tok_freqs2: ", self.output_tok_freqs)

        print("output_seqs: ",output_seqs)

        self.completed = True
        print(f"Pipeline completed for pipe {self.id}")
