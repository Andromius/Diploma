from filters.filter import Filter
from logging import Logger
from numpy import dot
from numpy.linalg import norm

class SimilarityFilter(Filter):
    def __init__(self, name: str = ["cosine", "euclidean", "hamming"], logger: Logger = None, threshold: float = 0.5):
        super().__init__("similarity", logger)
        self.threshold = threshold
        similarities = {
            "cosine": 0.0,
            "euclidean": 0.0,
            "hamming": 0.0
        }

    def apply(self, data: dict):
        x = data["fingerprint"]
        if "image_vectors" not in data:
            self.logger.info("No image vectors found in data.")
            return data
        else:
            for image in data["image_vectors"]:
                y = image["fingerprint"]
                for name in self.name:
                    self.similarities[name] = lambda x, y: dot(x, y) / (norm(x) * norm(y))

        self.logger.info(f"Applying {self.name} filter with threshold {self.threshold}")
        
        print(data)
        return data