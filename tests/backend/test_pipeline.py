import sys

sys.path.insert(0, '/Users/langston/Courses/cs435/csci-435_what_if_tool/src/codewit_semeru/backend')
from pipeline import Pipeline

def test_pipeline():
    pipe = Pipeline("gpt2", "Hello World") 
    assert pipe.model.config.output_attentions == True
    assert pipe.dataset == "Hello World"

def test_pipeline_run():
    pipe = Pipeline("gpt2", "Hello World")
    pipe.run()
    assert pipe.completed == True
    assert pipe.output_tkns #output tkns list not empty
