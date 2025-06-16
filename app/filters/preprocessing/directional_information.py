from filters.filter import Filter
from logging import Logger
import numpy as np
import cv2

class DirectionalInformationFilter(Filter):
    def __init__(self, logger: Logger):
        super().__init__(name=__name__, logger=logger)

    def apply(self, data: dict):
        if 'final_images' not in data or not isinstance(data['final_images'], list):
            self.logger.error("No final images found in data or final_images is not a list.")
            return data

        magnitudes = []
        directions = []

        for cutout in data['final_images']:
            if cutout.shape[2] == 3 or cutout.shape[2] == 4:
                gray = cv2.cvtColor(cutout, cv2.COLOR_RGB2GRAY)
            else:
                gray = cutout
                
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            magnitude = cv2.magnitude(grad_x, grad_y)
            direction = cv2.phase(grad_x, grad_y, angleInDegrees=True)
            # Normalize magnetude to range [0, 255]
            magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            magnitude = magnitude.astype(np.uint8)

            magnitudes.append(magnitude)
            directions.append(direction)

        data['magnitudes'] = magnitudes
        data['directions'] = directions

        return data