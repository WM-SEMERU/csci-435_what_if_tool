from .pipeline import Pipeline


class PipelineStore(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(PipelineStore, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        self.pipelines = []

    def add_pipeline(self, item: Pipeline) -> None:
        self.pipelines.insert(0, item)

    def remove_pipeline(self, index: int) -> None:
        if not self.size():
            return
        self.pipelines.pop(index)

    def get_pipeline(self, x: int) -> None:
        if x >= len(self.pipelines):
            return
        return self.pipelines[x]

    def run_pipelines(self) -> None:
        for pipe in self.pipelines:
            if not pipe.completed:
                pipe.run()

    def rerun_pipe(self, x: int) -> None:
        if x >= len(self.pipelines):
            return
        self.pipelines[x].run()

    def size(self) -> int:
        return len(self.pipelines)
