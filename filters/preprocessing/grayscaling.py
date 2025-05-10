import cv2
from logging import Logger
from filters.filter import Filter

class GrayscaleFilter(Filter):
    def __init__(self, logger: Logger):
        super().__init__(name="Grayscale", logger=logger)
    
    def apply(self, image):
        if len(image.shape) == 2:
            return image # already grayscaled
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
