from filters.model_segmentation_filter import ModelSegmentationFilter
from logging import Logger
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import torchvision.transforms as T
import torch

class MaskRCNNSegmentationFilter(ModelSegmentationFilter):
    def __init__(self, name : str, model_file : str, logger : Logger, resources_path: str, num_classes : int = 2):
        super().__init__(name, model_file, logger, resources_path)
        self.num_classes = num_classes
        self.model = self.load_model()

    def load_model(self):
        model = maskrcnn_resnet50_fpn(weights='DEFAULT')
        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, self.num_classes)

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, self.num_classes)

        model.load_state_dict(torch.load(self.resources_dir, map_location=self.device))
        model.to(self.device)
        model.eval()

        return model