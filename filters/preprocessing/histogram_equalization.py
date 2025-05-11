import cv2
from logging import Logger
from filters.filter import Filter

class HistogramEqualizationFilter(Filter):
    def __init__(self, logger: Logger):
        super().__init__(name="HistogramEqualization", logger=logger)

    def apply(self, data: dict):
        if 'image' not in data:
            self.logger.error("No image found in data.")
            raise ValueError("No image found in data.")

        if len(data['image'].shape) == 2:
            self.logger.info("Applying histogram equalization to grayscale image.")
            data['hist_equalization'] = cv2.equalizeHist(data['image'])
            return data

        elif len(data['image'].shape) == 3 and data['image'].shape[2] == 3:
            self.logger.info("Applying histogram equalization to color image (YCrCb space).")
            ycrcb = cv2.cvtColor(data['image'], cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            data['hist_equalization'] = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            return data

        else:
            self.logger.error("Unsupported image format for histogram equalization.")
            raise ValueError("Unsupported image format for histogram equalization.")