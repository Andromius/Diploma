from filters.model_segmentation_filter import ModelSegmentationFilter
from logging import Logger


class FastRCNNSegmentationFilter(ModelSegmentationFilter):
    def __init__(self, name : str, logger : Logger):
        super().__init__(name, logger)
