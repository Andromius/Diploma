from filters.model_segmentation_filter import ModelSegmentationFilter
from logging import Logger
import torch

class YoloSegmentationFilter(ModelSegmentationFilter):
    def __init__(self, name : str, model_file : str, logger : Logger):
        super().__init__(name, model_file, logger)
        self.model = self.load_model()

    def load_model(self):
        pass
        #return torch.hub.load('ultralytics/yolov5', 'yolov5s')