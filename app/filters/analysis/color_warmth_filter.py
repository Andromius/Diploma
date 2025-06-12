from filters.filter import Filter
import cv2
from logging import Logger
import numpy as np

class ColorWarmthFilter(Filter):
    def __init__(self, logger: Logger, chroma_threshold: int = 10):
        super().__init__(name="ColorWarmthFilter", logger=logger)
        self.chroma_threshold = chroma_threshold

    def apply(self, data: dict):
        if 'image' not in data:
            self.logger.error("No image found in data.")
            raise ValueError("No image found in data.")
        
        if 'segmentation_data' not in data:
            self.logger.error("No segmentation data found in data.")
            raise ValueError("No segmentation data found in data.")

        image = data['image']
        prediction = data['segmentation_data'][0]
        masks = prediction['masks'].cpu().numpy()  # Get predicted masks
        self.logger.info(f"Applying color warmth filter")

        data['warmth_scores'] = []
        for mask in masks:
            binary_mask = mask[0, :, :] > 0.8
            masked_img = image * binary_mask[:, :, np.newaxis]
            lab_img = cv2.cvtColor(masked_img.astype(np.uint8), cv2.COLOR_BGR2LAB)

            l_channel, a_channel, b_channel = cv2.split(lab_img)

            centered_a = a_channel - 128
            centered_b = b_channel - 128
            chroma = np.sqrt(centered_a**2 + centered_b**2)

            relevant_mask = (l_channel > 0) & (chroma > self.chroma_threshold)

            relevant_a = a_channel[relevant_mask]

            if relevant_a.size == 0:
                data['warmth_scores'].append(np.float64(0.5))
                continue

            data['warmth_scores'].append(np.mean(relevant_a) / 255)

        self.logger.info(f"Color warmth scores calculated: {data['warmth_scores']}")
        return data