import cv2
from logging import Logger
from filters.filter import Filter

class LineSeparationFilter(Filter):
    def __init__(self, logger: Logger):
        super().__init__(name="Contouring", logger=logger)

    def apply(self, data: dict):
        data["cutout_edges"] = []
        data["cutout_contours"] = []

        if "final_images" not in data or not isinstance(data["final_images"], list):
            raise ValueError("No final images found in data or final_images is not a list.")
        
        for cutout in data["final_images"]:
            if cutout.shape[2] == 4:
                cutout_gray = cv2.cvtColor(cutout[..., :3], cv2.COLOR_RGB2GRAY)
            elif cutout.shape[2] == 3:
                cutout_gray = cv2.cvtColor(cutout, cv2.COLOR_RGB2GRAY)
            else:
                cutout_gray = cutout

            edges = cv2.Canny(cutout_gray, 100, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            data["cutout_edges"].append(edges)
            data["cutout_contours"].append(contours)

        self.logger.info(f"Applied line separation filter. Found {len(data['cutout_edges'])} edges and {len(data['cutout_contours'])} contours.")
        
        return data