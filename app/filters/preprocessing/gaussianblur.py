from filters.filter import Filter
from logging import Logger
import numpy as np
import torch
import cv2

class GaussianBlurFilter(Filter):
    def __init__(self, logger: Logger, kernel_size: tuple = (5, 5), sigma: float = 0.65):
        super().__init__(name="GaussianBlur", logger=logger)
        self.kernel_size = kernel_size
        self.sigma = sigma
        
    def apply(self, data: dict):       
        if 'final_images' not in data or not isinstance(data['final_images'], list):
            raise ValueError("No final images found in data or final_images is not a list.")
        
        self.logger.info(f"Size of final images: {len(data['final_images'])}")
        
        blurred_cutouts = []
        for cutout in data['final_images']:
            blurred_cutout = cv2.GaussianBlur(cutout, self.kernel_size, self.sigma)
            blurred_cutouts.append(blurred_cutout)

        data['final_images'] = blurred_cutouts
    
        self.logger.info(f"Applied Gaussian blur with kernel {self.kernel_size} and sigma {self.sigma} and size {len(data['final_images'])}.")
        return data
