from filters.filter import Filter
from logging import Logger
import numpy as np
import cv2

class MorphologicalGradientFilter(Filter):
    def __init__(self, logger: Logger, kernel_size=(9, 9)):
        super().__init__(name=__name__, logger=logger)
        self.kernel_size = kernel_size

    def apply(self, data: dict):
        if 'final_images' not in data or not isinstance(data['final_images'], list):
            self.logger.error("No final images found in data or final_images is not a list.")
            return data

        kernel = np.ones(self.kernel_size, np.uint8)

        gradient_images = []
        for cutout in data['final_images']:
            # Apply morphological gradient
            gradient = cv2.morphologyEx(cutout, cv2.MORPH_GRADIENT, kernel)
            enhanced_gradient = cv2.convertScaleAbs(gradient, alpha=5, beta=0)
            #_, binary_edges = cv2.threshold(enhanced_gradient, 10, 255, cv2.THRESH_BINARY)

            gradient_images.append(gradient)

        data['gradients'] = gradient_images
        return data