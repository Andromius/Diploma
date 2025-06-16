import cv2
from logging import Logger
from filters.filter import Filter
import numpy as np

class LineSeparationFilter(Filter):
    def __init__(self, logger: Logger):
        super().__init__(name="Contouring", logger=logger)

    def apply(self, data: dict):
        data["cutout_edges"] = []
        data["cutout_contours"] = []
        data["hierarchies"] = []

        if "final_images" not in data or not isinstance(data["final_images"], list):
            raise ValueError("No final images found in data or final_images is not a list.")
        if "thresh" not in data or len(data["thresh"]) != len(data["final_images"]):
            raise ValueError("Thresholded images missing or mismatched length.")

        for cutout, thresh in zip(data["final_images"], data["thresh"]):
            # Convert cutout to grayscale for edge detection
            if cutout.shape[2] == 4:
                cutout_gray = cv2.cvtColor(cutout[..., :3], cv2.COLOR_RGB2GRAY)
            elif cutout.shape[2] == 3:
                cutout_gray = cv2.cvtColor(cutout, cv2.COLOR_RGB2GRAY)
            else:
                cutout_gray = cutout

            # Ensure thresh is a numpy array of type uint8 with one channel
            if isinstance(thresh, list):
                thresh = np.array(thresh, dtype=np.uint8)
            if len(thresh.shape) > 2:
                thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

            # Apply Canny edge detection
            edges = cv2.Canny(cutout_gray, 100, 200)

            # Find contours on threshold image
            cutout_copy = cutout.copy()
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            #cutout_contours = cv2.drawContours(cutout_copy, contours, -1, (0, 255, 0), thickness=3)


            data["cutout_edges"].append(edges)
            data["cutout_contours"].append(contours)
            data["hierarchies"].append(hierarchy)

        self.logger.info(f"Applied line separation filter. Found {len(data['cutout_edges'])} edge maps and {len(data['cutout_contours'])} contour sets.")
        return data