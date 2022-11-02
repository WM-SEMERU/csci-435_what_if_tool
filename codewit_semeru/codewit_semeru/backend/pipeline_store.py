from .pipeline import Pipeline

class PipelineStore(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(PipelineStore, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        self.pipelines = []

    def addPipeline(self, item: Pipeline) -> None:
        self.pipelines.append(item)

    def getPipeline(self, x: int) -> None:
        return self.pipelines[x]

    def runPipelines(self) -> None:
        for pipe in self.pipelines:
            pipe.start()

    def rerunPipe(self, x: int) -> None:
        if x >= len(self.pipelines):
            return
        self.pipelines[x].start()


