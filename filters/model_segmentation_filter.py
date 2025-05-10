from logging import Logger
from filter import Filter

class ModelSegmentationFilter(Filter):
    def __init__(self, name: str, logger: Logger):
        super().__init__(name, logger)
        self.name = name
        self.logger = logger

    def load_model(self):
        raise NotImplementedError("Subclasses should implement this method.")