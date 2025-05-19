from filters.filter import Filter
from logging import Logger
import numpy as np
import torch
import cv2

class GaussianBlurFilter(Filter):
    def __init__(self, logger: Logger, kernel_size: tuple = (10, 10), sigma: float = 3.0):
        super().__init__(name="GaussianBlur", logger=logger)
        self.kernel_size = kernel_size
        self.sigma = sigma
        
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

        blurred_masks = []
        for mask in masks:
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0, :, :]

            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)

            mask_blur = cv2.GaussianBlur(mask, self.kernel_size, self.sigma)
            blurred_masks.append(mask_blur)

        blurred_masks_array = np.array(blurred_masks)
        if len(original_shape) == 4:
            blurred_masks_array = blurred_masks_array.reshape(original_shape[0], 1, original_shape[2], original_shape[3])
        elif len(original_shape) == 3:
            blurred_masks_array = blurred_masks_array.reshape(1, original_shape[1], original_shape[2])

        prediction['masks'] = torch.from_numpy(blurred_masks_array).to(original_device).to(original_dtype)
        data['segmentation_data'][0] = prediction

        self.logger.info(f"Applied Gaussian blur with kernel {self.kernel_size} and sigma {self.sigma}.")
        return data
