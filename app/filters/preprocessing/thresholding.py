from filters.filter import Filter
from logging import Logger
import numpy as np
import torch
import cv2

class ThresholdingFilter(Filter):
    def __init__(self, logger: Logger, threshold: float = 0.8):
        super().__init__(name="Thresholding", logger=logger)
        self.threshold = threshold

    def apply(self, data: dict):
        if 'segmentation_data' not in data:
            raise ValueError("No segmentation data found in data.")
        if not isinstance(data['segmentation_data'], list):
            raise ValueError("Segmentation data is not in the expected format (expected list).")

        prediction = data["segmentation_data"][0]
        masks = prediction['masks'].cpu().numpy()
        original_dtype = prediction['masks'].dtype
        original_device = prediction['masks'].device
        original_shape = prediction['masks'].shape

        thresholded_masks = []
        for mask in masks:
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0, :, :]

            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)

            _, mask_thresh = cv2.threshold(mask, int(self.threshold * 255), 255, cv2.THRESH_BINARY)
            
            thresholded_masks.append(mask_thresh)

        thresholded_masks_array = np.array(thresholded_masks)
        if len(original_shape) == 4:
            thresholded_masks_array = thresholded_masks_array.reshape(original_shape[0], 1, original_shape[2], original_shape[3])
        elif len(original_shape) == 3:
            thresholded_masks_array = thresholded_masks_array.reshape(1, original_shape[1], original_shape[2])

        prediction['masks'] = torch.from_numpy(thresholded_masks_array).to(original_device).to(original_dtype)
        data['segmentation_data'][0] = prediction

        self.logger.info(f"Applied thresholding filter with threshold {self.threshold}.")
        return data