import sys

sys.path.insert(0, '/Users/langston/Courses/cs435/csci-435_what_if_tool/src/codewit_semeru/backend')
from pipeline_store import PipelineStore
from pipeline import Pipeline


def test_add_pipeline():
    store = PipelineStore()
    pipe = Pipeline("gpt2", "Hello World")
    assert not store.pipelines #is empty

    store.add_pipeline(pipe)
    assert len(store.pipelines) == 1 #has one pipeline

def test_remove_existing_pipeline():
    store = PipelineStore()
    pipe = Pipeline("gpt2", "Hello World")

    assert not store.pipelines #is empty
    store.add_pipeline(pipe)
    assert len(store.pipelines) == 1 #has one pipeline
    store.remove_pipeline(0)
    assert not store.pipelines #is empty

def test_remove_non_existing_pipeline():
    store = PipelineStore()
    pipe = Pipeline("gpt2", "Hello World")

    assert not store.pipelines #is empty
    store.remove_pipeline(0)
    assert not store.pipelines #is empty

def test_get_existing_pipeline():
    store = PipelineStore()
    pipe = Pipeline("gpt2", "Hello World")

    store.add_pipeline(pipe)
    assert len(store.pipelines) == 1 #has one pipeline

    pipe1 = store.get_pipeline(0)

    assert pipe.model == pipe1.model
    assert pipe.dataset == pipe1.dataset
    assert pipe.input_tkns == pipe1.input_tkns

def test_get_non_existing_pipeline():
    store = PipelineStore()
    pipe = Pipeline("gpt2", "Hello World")

    pipe1 = store.get_pipeline(0)

    assert pipe1 == None


def test_run_existing_pipeline():
    store = PipelineStore()
    pipe = Pipeline("gpt2", "Hello World")

    store.add_pipeline(pipe)
    store.run_pipelines()

    assert store.get_pipeline(0).completed == True

def test_rerun_pipe():
    store = PipelineStore()
    pipe = Pipeline("gpt2", "Hello World")

    store.add_pipeline(pipe)
    store.run_pipelines()
    assert store.get_pipeline(0).completed == True

    store.rerun_pipe(0)
    assert store.get_pipeline(0).completed == True

def test_rerun_non_existing_pipe():
    store = PipelineStore()
    pipe = Pipeline("gpt2", "Hello World")

    assert store.rerun_pipe(0) == None

def test_size():
    store = PipelineStore()
    pipe = Pipeline("gpt2", "Hello World")
    #assert store.size == 0
    print(store.size)
    store.add_pipeline(pipe)   
    print(store.size)
    assert store.size == 1

    store.remove_pipeline(0)
    assert store.size == 0


