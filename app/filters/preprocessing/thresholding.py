from filters.filter import Filter
from logging import Logger
import numpy as np
import cv2

class ThresholdingFilter(Filter):
    def __init__(self, logger: Logger, threshold: float = 0.8):
        super().__init__(name="Thresholding", logger=logger)
        self.threshold = threshold

    def apply(self, data: dict):
        if 'final_images' not in data or not isinstance(data['final_images'], list):
            self.logger.error("No final images found in data or final_images is not a list.")
            raise ValueError("No final images found in data or final_images is not a list.")
        
        thresholded_cutouts = []
        for cutout in data['final_images']:
            if cutout.shape[2] == 4:
                cutout_gray = cv2.cvtColor(cutout[..., :3], cv2.COLOR_RGB2GRAY)
            elif cutout.shape[2] == 3:
                cutout_gray = cv2.cvtColor(cutout, cv2.COLOR_RGB2GRAY)
            else:
                cutout_gray = cutout
            
            # Apply thresholding
            thresholded_cutout = cv2.adaptiveThreshold(cutout_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)
            thresholded_cutouts.append(thresholded_cutout)

        data['thresh'] = thresholded_cutouts
        self.logger.info(f"Applied thresholding with threshold value {self.threshold}.")
        return data