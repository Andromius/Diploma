import cv2
from logging import Logger
from filters.filter import Filter

class GaussianBlurFilter(Filter):
    def __init__(self, logger: Logger, kernel_size: tuple = (5, 5), sigma: float = 0):
        super().__init__(name="GaussianBlur", logger=logger)
        self.kernel_size = kernel_size
        self.sigma = sigma
        
    def apply(self, image):
        self.logger.info(f"Applying Gaussian blur with kernel size {self.kernel_size} and sigma {self.sigma}.")
        return cv2.GaussianBlur(image, self.kernel_size, self.sigma)
