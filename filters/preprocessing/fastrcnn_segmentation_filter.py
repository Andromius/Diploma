from filters.model_segmentation_filter import ModelSegmentationFilter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from logging import Logger
import torch


class FastRCNNSegmentationFilter(ModelSegmentationFilter):
    def __init__(self, name : str, logger : Logger, num_classes : int = 2):
        super().__init__(name, logger)
        self.num_classes = num_classes
        self.model = self.load_model()

    def load_model(self):
        model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, self.num_classes)

        model.load_state_dict(torch.load(self.resources_dir))
        model.to(self.device)
        model.eval()

        return model