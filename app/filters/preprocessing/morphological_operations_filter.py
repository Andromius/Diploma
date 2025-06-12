from filters.filter import Filter
from logging import Logger
import numpy as np
import torch
import cv2

class MorphologicalOperationsFilter(Filter):
    def __init__(self, logger: Logger, kernel_size=(3, 3), iterations=5):
        super().__init__(name=__name__, logger=logger)
        self.kernel_size = kernel_size
        self.iterations = iterations

    def apply(self, data: dict):
        """
        Applies morphological operations to the segmentation masks in the input data.

        Args:
            data (dict): A dictionary containing 'segmentation_data'.
                   'segmentation_data' is expected to be a list, where the first element
                   contains the mask.

        Returns:
            dict: The modified data dictionary, with cleaned masks in the 'segmentation_data'.
                    The 'segmentation_data'  will be updated
        Raises:
            ValueError: If 'segmentation_data' is missing or has the wrong type.
        """
        if 'segmentation_data' not in data:
            raise ValueError("No segmentation data found in data.")
        if not isinstance(data['segmentation_data'], list):
            raise ValueError("Segmentation data is not in the expected format (expected list).")

        prediction = data["segmentation_data"][0]  # Get the first prediction
        masks = prediction['masks'].cpu().numpy()  # Get predicted masks as numpy array
        original_dtype = prediction['masks'].dtype  # Get the original data type
        original_device = prediction['masks'].device  # Get the original device
        original_shape = prediction['masks'].shape # Get the original shape


        # Define the structuring element (kernel) for morphological operations.
        kernel = np.ones(self.kernel_size, np.uint8)

        cleaned_masks = []

        # Iterate through each detected object
        for mask in masks:
            # Convert the mask to a binary mask (0 or 1)
            binary_mask = (mask[0, :, :] > 0.5).astype(np.uint8)

            # Apply morphological operations: Erosion followed by Dilation (Opening)
            eroded_mask = cv2.erode(binary_mask, kernel, iterations=self.iterations)
            dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=self.iterations)
            cleaned_masks.append(dilated_mask)

        # Update the masks in the prediction data with the cleaned masks, preserving original shape
        cleaned_masks_array = np.array(cleaned_masks)
        if len(original_shape) == 4:
            cleaned_masks_array = cleaned_masks_array.reshape(original_shape[0], 1, original_shape[2], original_shape[3])
        elif len(original_shape) == 3:
             cleaned_masks_array = cleaned_masks_array.reshape(1, original_shape[1], original_shape[2])

        prediction['masks'] = torch.from_numpy(cleaned_masks_array).to(original_device).to(original_dtype)
        data['segmentation_data'][0] = prediction  # update the first element of the list

        return data