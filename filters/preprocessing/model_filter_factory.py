from filters.preprocessing.yolo_segmentation_filter import YoloSegmentationFilter
from filters.preprocessing.fastrcnn_segmentation_filter import FastRCNNSegmentationFilter
from filters.preprocessing.maskrcnn_segmentation_filter import MaskRCNNSegmentationFilter
from logging import Logger

class ModelFilterFactory:
    def __init__(self, logger : Logger):
        self.model_paths = {"yolo" : "yolo.pt", "maskRCNN" : "maskrcnn.pth", "fastRCNN" : None}
        self.logger = logger

    def create_yolo(self):
        return YoloSegmentationFilter("yolo", self.logger)
    
    def create_maskRCNN(self):
        return MaskRCNNSegmentationFilter("maskRCNN", self.logger)
    
    def create_fastRCNN(self):
        return FastRCNNSegmentationFilter("fastRCNN", self.logger)