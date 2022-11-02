from .pipeline import Pipeline

class PipelineStore(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(PipelineStore, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        self.pipelines = []

    def addPipeline(self, item: Pipeline) -> None:
        self.pipelines.insert(0, item)
    
    def removePipeline(self, index:int) -> None:
        if not self.size():
            return
        self.pipelines.pop(index)

    def getPipeline(self, x: int) -> None:
        return self.pipelines[x]

    def runPipelines(self) -> None:
        for pipe in self.pipelines:
            if not pipe.completed:
                pipe.start()

    def rerunPipe(self, x: int) -> None:
        if x >= len(self.pipelines):
            return
        self.pipelines[x].start()
    
    def size(self) -> int:
        return len(self.pipelines)


