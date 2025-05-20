from filters.filter import Filter
from logging import Logger
import numpy as np

class OutputFilter(Filter):
    def __init__(self, logger : Logger, name : str = "output"):
        super().__init__(name, logger)

    def apply(self, data : dict):
        prediction = data["segmentation_data"]
        image = data["image"]
        masks = prediction[0]['masks'].cpu().numpy()  # Get predicted masks
        labels = prediction[0]['labels'].cpu().numpy()  # Get predicted labels
        scores = prediction[0]['scores'].cpu().numpy()  # Get predicted scores

        arr = list(zip(masks, labels, scores))

        images = []
        
        for i, (mask, label, score) in enumerate(arr):
            binary_mask = mask[0, :, :] > 0.5

            # Get the bounding box of the mask
            y_indices, x_indices = np.where(binary_mask)
            if len(x_indices) > 0 and len(y_indices) > 0:
                # Get shape of the mask
                shape_mask = np.zeros_like(binary_mask, dtype=bool)
                shape_mask[y_indices, x_indices] = True
                
                # Extract just those pixels from the original image
                # We need to determine what pixels to include
                unique_y = sorted(set(y_indices))
                unique_x = sorted(set(x_indices))
                
                # Create a mapping from original coordinates to our new compact image
                y_map = {y: i for i, y in enumerate(unique_y)}
                x_map = {x: i for i, x in enumerate(unique_x)}
                
                # Create a new image with just the required size
                new_h, new_w = len(unique_y), len(unique_x)
                if len(image.shape) == 3 and image.shape[2] == 3:  # RGB
                    cutout = np.zeros((new_h, new_w, 4), dtype=np.uint8)
                    
                    # Place only the masked pixels into the new image
                    for orig_y, orig_x in zip(y_indices, x_indices):
                        new_y, new_x = y_map[orig_y], x_map[orig_x]
                        cutout[new_y, new_x, :3] = image[orig_y, orig_x]
                        cutout[new_y, new_x, 3] = 255
                else:  # Grayscale
                    cutout = np.zeros((new_h, new_w, 4), dtype=np.uint8)
                    for orig_y, orig_x in zip(y_indices, x_indices):
                        new_y, new_x = y_map[orig_y], x_map[orig_x]
                        cutout[new_y, new_x, :3] = image[orig_y, orig_x]
                        cutout[new_y, new_x, 3] = 0
                
                images.append(cutout)

        data["final_images"] = images
        return data