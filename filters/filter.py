from logging import Logger

class Filter:
    def __init__(self, name: str, logger: Logger):
        self.name = name
        self.logger = logger

    def apply(self, image):
        raise NotImplementedError("Subclasses should implement this method.")