import cv2
from logging import Logger
from filters.filter import Filter

class ContouringFilter(Filter):
    def __init__(self, logger: Logger):
        super().__init__(name="Contouring", logger=logger)
    
    def apply(self,image):
        # edge detection
        self.logger.info("Contouring Filter: Edge detection")
        edges = cv2.Canny(image, 50, 150)
        self.logger.info("Contouring Filter: Finding contours...")
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # apply contours on original image (delete?)
        output = image.copy()
        cv2.drawContours(output, contours, -1, (0,255,0), 2) # (0,255,0) green contour
        self.logger.info("Contouring Filter: Done!")
        return contours
