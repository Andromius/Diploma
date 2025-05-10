class Filter:
    def __init__(self, name: str):
        self.name = name

    def apply(self, image):
        raise NotImplementedError("Subclasses should implement this method.")