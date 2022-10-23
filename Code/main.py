import sys, argparse
from run_model import run_pipeline


def WITCode(model, dataset, tokenizer):
    print("test")
    run_pipeline(model, dataset, tokenizer)



if __name__ == "__main__":
    WITCode("gpt2", "This is some chunk of code that I wish to analyze", "gpt2")