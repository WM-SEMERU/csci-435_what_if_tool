import sys
import os
import json
import requests
import time
import re
import ast
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

        self.output_seqs = []
        self.output_tok_freqs = defaultdict(list)

        self.completed: bool = False

    def query_model(self):
        print("Querying HF Inference API, this will take a moment...")
        data = json.dumps({"inputs": self.dataset, "parameters": {"return_full_text": False, "max_new_tokens": 50, "max_time": 30}})
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

        self.output_seqs = [data[0]["generated_text"] for data in res]
        output_tkns = [self.tokenizer.tokenize(seq) for seq in self.output_seqs]

        for tkns in output_tkns:
            cts = Counter(tkns)
            for tkn in cts:
                self.output_tok_freqs[tkn].append(cts[tkn])

        # add 0 freq counts for tokens which were not within all predicted sequences
        for tkn in self.output_tok_freqs:
            seq_diff = len(output_tkns) - len(self.output_tok_freqs[tkn])
            self.output_tok_freqs[tkn].extend([0] * seq_diff)                
        
        self.completed = True
        print(f"Pipeline completed for pipe {self.id}")

        self.parse_code_concepts()


    def convert_to_python_code(self, output_sequence: str) -> str:
        # Define a regular expression to match indentation
        # indent_regex = re.compile(r":\s*$")

        indent_regex = re.compile(r"^(def|class|if|try) ")
        unindent_regex = re.compile(r"^(elif|else|except) ")

        return_regex = re.compile(r"^return ")

        # Split the output sequence into individual lines
        lines = output_sequence.strip().split("<EOL>")

        # Define a regular expression to match and remove spaces that are not part of keywords or string literals
        space_regex = re.compile(r"(?<![\'\"])(?<!\b)\s+(?![\'\"])")

        indent_level = 0

        formatted_code = ""

        for line in lines:
            # Replace spaces with the empty string
            line = space_regex.sub("", line)

            # Remove the <s> delimiters from the line
            line = line.replace("<s>", "")

            if unindent_regex.match(line):
                indent_level -= 4
                formatted_code += " " * indent_level + line + "\n"
                indent_level += 4
            else:
                formatted_code += " " * indent_level + line + "\n"
                if indent_regex.match(line):
                    indent_level += 4
                elif return_regex.match(line):
                    indent_level -= 4

        print(formatted_code)

        return formatted_code


    def parse_code_concepts(self):
        print(self.dataset[1] + self.output_seqs[1])
        tree = ast.parse(self.convert_to_python_code(self.dataset[1] + self.output_seqs[1]))

        # Initialize counters for each type of node or expression
        function_body_types = {
            "function call": 0,
            "if statement": 0,
            "for loop": 0,
            "compound statement": 0,
        }
        if_statements = 0
        function_calls = 0
        print_statements = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Collect the nodes for the function body
                body = []
                for subnode in ast.walk(node):
                    if isinstance(subnode, (ast.Expr, ast.Assign, ast.If, ast.For)):
                        body.append(subnode)
                # Determine the type of the function body and increment the corresponding counter
                if len(body) == 1 and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Call):
                    function_body_types["function call"] += 1
                elif len(body) == 1 and isinstance(body[0], ast.If):
                    function_body_types["if statement"] += 1
                elif len(body) == 1 and isinstance(body[0], ast.For):
                    function_body_types["for loop"] += 1
                else:
                    function_body_types["compound statement"] += 1
            elif isinstance(node, ast.If):
                if_statements += 1
            elif isinstance(node, ast.Call):
                function_calls += 1

        # Print the counts for each type of node or expression
        print("function body types:")
        for key, value in function_body_types.items():
            print("  ", key, ":", value)

        print(f"if statements: {if_statements}")
        print(f"function calls: {function_calls}")
        print(f"print_statements: {print_statements}")