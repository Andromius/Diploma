import cv2
from logging import Logger
from filters.filter import Filter

class GrayscaleFilter(Filter):
    def __init__(self, logger: Logger):
        super().__init__(name="Grayscale", logger=logger)
    
    def apply(self, data: dict):
        if 'image' not in data:
            self.logger.error("No image found in data.")
            raise ValueError("No image found in data.")
        image = data['image']

        if len(image.shape) == 2:
            return data # already grayscaled
        
        data['grayscale'] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return data
