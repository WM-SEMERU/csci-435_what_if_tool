from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Pipeline:
    def __init__(self, tokenizer, model, dataset) -> None:
        # to-do https://github.com/tensorflow/tensorflow/issues/53529
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(model, output_attentions=True).to(self.device)
        self.dataset = dataset
        self.output = None
        self.attention = None
        self.output_strs = None
        self.output_tkns = None
        self.device = None

        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.input_ids = self.tokenizer(dataset, return_tensors="pt").input_ids.to(self.device)
        self.input_tkns = self.tokenizer.convert_ids_to_tokens(self.input_ids[0])

    def start(self) -> None:
        self.output = self.model.generate(self.input_ids, do_sample=False, max_length=50)
        self.attention = self.output[-1]
        self.output_strs = self.tokenizer.batch_decode(self.output, skip_special_tokens=True)
        self.output_tkns = self.tokenizer.tokenize(self.output_strs[0])
        print("Completed")
