from filters.filter import Filter
from logging import Logger
import numpy as np
import torch
import cv2

class MorphologicalOperationsFilter(Filter):
    def __init__(self, logger: Logger, kernel_size=(3, 3)):
        super().__init__(name=__name__, logger=logger)
        self.kernel_size = kernel_size

    def apply(self, data: dict):
        if 'final_images' not in data or not isinstance(data['final_images'], list):
            self.logger.error("No final images found in data or final_images is not a list.")

        # Define the structuring element (kernel) for morphological operations.
        kernel = np.ones(self.kernel_size, np.uint8)

        cleaned_cutouts = []
        for cutout in data['final_images']:
            # Erosion + dilatation to remove noise
            cutout_open = cv2.morphologyEx(cutout, cv2.MORPH_OPEN, kernel)

            # Dilatation + erosion to fill small holes
            cutout_close = cv2.morphologyEx(cutout_open, cv2.MORPH_CLOSE, kernel)
            cleaned_cutouts.append(cutout_close)

        data['final_images'] = cleaned_cutouts
        return data