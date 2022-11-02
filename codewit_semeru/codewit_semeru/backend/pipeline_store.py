class PipelineStore(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(PipelineStore, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        self.pipelines = []

    def addPipeline(self, item):
        self.pipelines.append(item)

a = PipelineStore()
b = PipelineStore()

print (a is b)

a.addPipeline(4)
b.addPipeline(37)

print(a.pipelines)
print(b.pipelines)

