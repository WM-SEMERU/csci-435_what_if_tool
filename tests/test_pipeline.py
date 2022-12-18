
from codewit_semeru.backend.pipeline import Pipeline

def test_pipeline():
    pipe = Pipeline("gpt2", "Hello World", "test") 

    assert pipe.API_URL == f"https://api-inference.huggingface.co/models/gpt2"
    assert pipe.id == "gpt2<>test"
    assert pipe.model == "gpt2"
    assert pipe.dataset == "Hello World"
    assert pipe.dataset_id == "test"
    assert not pipe.output
    assert not pipe.output_strs
    assert not pipe.output_tkns
    assert not pipe.output_tok_freqs
    assert not pipe.completed

def test_pipeline_run():
    pipe = Pipeline("gpt2", "Hello World")
    pipe.run()
    print(pipe.output_tkns)
    assert pipe.completed == True
    assert pipe.output_strs #output str list not empty
