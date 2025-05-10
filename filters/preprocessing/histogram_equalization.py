import cv2
from logging import Logger
from filters.filter import Filter

class HistogramEqualizationFilter(Filter):
    def __init__(self, logger: Logger):
        super().__init__(name="HistogramEqualization", logger=logger)

    def apply(self, image):
        if len(image.shape) == 2:
            self.logger.info("Applying histogram equalization to grayscale image.")
            return cv2.equalizeHist(image)

        elif len(image.shape) == 3 and image.shape[2] == 3:
            self.logger.info("Applying histogram equalization to color image (YCrCb space).")
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        else:
            self.logger.error("Unsupported image format for histogram equalization.")
            raise ValueError("Unsupported image format for histogram equalization.")