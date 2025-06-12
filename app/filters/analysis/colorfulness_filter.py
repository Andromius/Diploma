from filters.filter import Filter
from logging import Logger
import numpy as np

class ColorfulnessFilter(Filter):
    def __init__(self, logger: Logger):
        super().__init__(__name__, logger)

    def apply(self, data: dict):
        if 'segmentation_data' not in data:
            raise ValueError("No segmentation data found in data.")
        if 'image' not in data:
            raise ValueError("No image found in data.")
        if type(data['segmentation_data']) is not list:
            raise ValueError("Segmentation data is not in the expected format.")
        
        self.logger.info("Applying colorfulness filter.")

        prediction = data["segmentation_data"][0]
        image = data["image"]
        masks = prediction['masks'].cpu().numpy()  # Get predicted masks

        colorfulnesses = []
        
        for mask in masks:
            binary_mask = mask[0, :, :] > 0.8
            # Extract RGB channels for the masked region
            masked_img = image * binary_mask[:, :, np.newaxis]
            
            non_black_mask = ~np.all(masked_img == [0, 0, 0], axis=2)
            masked_img = masked_img[non_black_mask]

            b, g, r = masked_img[:,0], masked_img[:,1], masked_img[:,2]

            rg = r - g
            yb = 0.5 * (r + g) - b

            rg_mean = np.mean(rg)
            yb_mean = np.mean(yb)
            rg_std = np.std(rg)
            yb_std = np.std(yb)

            colorfulness = np.sqrt(rg_std**2 + yb_std**2) + 0.3 * np.sqrt(rg_mean**2 + yb_mean**2)
            colorfulnesses.append(colorfulness / 300)
        
        data['colorfulness_data'] = colorfulnesses
        return data
