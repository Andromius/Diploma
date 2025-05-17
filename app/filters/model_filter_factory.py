from filters.preprocessing.yolo_segmentation_filter import YoloSegmentationFilter
from filters.preprocessing.fastrcnn_segmentation_filter import FastRCNNSegmentationFilter
from filters.preprocessing.maskrcnn_segmentation_filter import MaskRCNNSegmentationFilter
from logging import Logger

class ModelFilterFactory:
    def __init__(self, logger : Logger, resources_path: str):
        self.model_paths = {"yolo" : "yolo.pt", "maskRCNN" : "maskrcnn.pth", "fastRCNN" : None}
        self.logger = logger
        self.resources_path = resources_path

    def create_model(self, name):
        if name not in self.model_paths:
            raise ValueError(f"Model {name} not found. Available models: {list(self.model_paths.keys())}")
        
        if name == "yolo":
            return YoloSegmentationFilter(name, self.model_paths[name], self.logger, self.resources_path)
        elif name == "maskRCNN":
            return MaskRCNNSegmentationFilter(name, self.model_paths[name], self.logger, self.resources_path)
        elif name == "fastRCNN":
            return FastRCNNSegmentationFilter(name, self.model_paths[name], self.logger, self.resources_path)
        else:
            raise ValueError(f"Model {name} not supported.")