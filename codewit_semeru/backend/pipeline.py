from collections import Counter, defaultdict
from typing import List
from uuid import uuid4
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os, json, requests
from .config import config

class Pipeline:
    # to-do https://github.com/tensorflow/tensorflow/issues/53529
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def pipe_id(model: str, dataset_id: str) -> str:
        if dataset_id == None:
            dataset_id = str(uuid4())
        return "<>".join([model, dataset_id])

    def __init__(self, model: str, dataset: List[str], dataset_id: str = None) -> None:
        self.HF_API_KEY = config.HF_API_TOKEN

        self.API_URL = f"https://api-inference.huggingface.co/models/{model}"
        self.headers = {"Authorization": f"Bearer {self.HF_API_KEY}"}

        if dataset_id == None:
            dataset_id = str(uuid4())
        self.id: str = Pipeline.pipe_id(model, dataset_id)

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = model
        self.dataset: List[str] = dataset
        self.dataset_id = dataset_id

        self.output = []
        
        self.output_strs: List[str] = []
        self.output_tkns: List[str] = []
        self.output_tok_freqs = defaultdict(list)

        self.completed: bool = False


    # TODO: Update so output doesn't contain input sequence!
    def run(self) -> None:
        # Weird interaction here where specifiying transformers generate pipeline + getting attention does not quite work...
        # to-do : figure out how to extract all necessary info from one pipeline run
        def query(payload):
            data = json.dumps(payload)
            response = requests.request("POST", self.API_URL, headers=self.headers, data=data)
            return json.loads(response.content.decode("utf-8"))

        for i in range(len(self.dataset)):

            data = query(self.dataset[i])
            self.output_strs.append([data[0]['generated_text']])
            if self.model == "codeparrot/codeparrot-small":
                self.output_tkns.append(self.tokenizer.tokenize(self.output_strs[i][0]))
            # self.output_tkns.append(self.tokenizer.tokenize(self.output_strs[i]))
            print(data[0])
        print(f'Output tkns: {self.output_tkns}')         

        for tokens in self.output_tkns:
            counts = Counter(tokens)
            for token in counts:
                self.output_tok_freqs[token].append(counts[token])

        # Add 0 freq counts for tokens which were not within all predicted sequences
        for token in self.output_tok_freqs:
            for _ in range(len(self.output_tkns) - len(self.output_tok_freqs[token])):
                self.output_tok_freqs[token].append(0)

        self.completed = True
        print("output_strs: ", self.output_strs)
        print(f"Pipeline completed for pipe {self.id}")
    
    
