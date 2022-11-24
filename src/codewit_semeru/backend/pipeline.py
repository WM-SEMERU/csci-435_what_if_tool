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
        self.output = []  
        self.test_output = []
        self.attention = []
        self.output_strs = []
        self.output_tkns = []
        self.completed = False

        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.input_ids = []
        self.input_tkns = []
        for i in range(len(dataset)):
            self.input_ids.append(self.tokenizer(
                dataset[i], return_tensors="pt").input_ids.to(self.device))
            self.input_tkns.append(self.tokenizer.convert_ids_to_tokens(
                self.input_ids[i][0]))

    def run(self) -> None:
        # Weird interaction here where specifiying transformers generate pipeline + getting attention does not quite work...
        # to-do : figure out how to extract all necessary info from one pipeline run
        for i in range(len(self.dataset)):
            self.output.append(self.model.generate(
                self.input_ids[i], do_sample=False, max_length=50))
            self.test_output.append(self.model(self.input_ids[i]))
            self.attention.append(self.test_output[i][-1])
            self.output_strs.append(self.tokenizer.batch_decode(
                self.output[i], skip_special_tokens=True))
            self.output_tkns.append(self.tokenizer.tokenize(self.output_strs[i][0]))
        print('updated')
        print(self.output)

        self.completed = True 
        # print(self.attention)
        print(self.input_tkns)
        print("Completed")
