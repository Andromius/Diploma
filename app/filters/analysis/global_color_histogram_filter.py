from filters.filter import Filter
import cv2
from logging import Logger
import numpy as np

class GlobalColorHistogramFilter(Filter):
    def __init__(self, logger: Logger, bins: int = 8):
        super().__init__(name="GlobalColorHistogram", logger=logger)
        self.bins = bins

    def apply(self, data: dict):
        if 'image' not in data:
            self.logger.error("No image found in data.")
            raise ValueError("No image found in data.")

        image = data['image']
        if len(image.shape) != 3 or image.shape[2] != 3:
            self.logger.error("Image must be a 3-channel color image.")
            raise ValueError("Image must be a 3-channel color image.")
        
        prediction = data['segmentation_data'][0]
        masks = prediction['masks'].cpu().numpy()
        
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        histograms = []
        for mask in masks:
            binary_mask = (mask[0, :, :] > 0.8).astype(np.uint8) * 255
            self.logger.info("Calculating global color histogram.")
            hist = cv2.calcHist([image_lab], [0, 1, 2], binary_mask, [self.bins, self.bins, self.bins], [0, 255, 0, 255, 0, 255])
            hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1).flatten()
            self.logger.info(f"Histogram shape: {hist.shape}, Histogram sum: {np.sum(hist)}")
            histograms.append(hist)

        data['global_color_histograms'] = histograms
        return data