import pytest
import pandas as pd
from .test_filter import TestFilter
from app.filters.preprocessing.maskrcnn_segmentation_filter import MaskRCNNSegmentationFilter
import torch

class TestModelFilter(TestFilter):

    # def test_yolo_apply(self, test_yolo_image_dict, app):
    #     # Assuming the model has a method called 'predict' that takes an image and returns predictions
    #     filter = ModelSelectionFilter("yolo", app.logger)
    #     predictions = filter.apply(test_yolo_image_dict)
    #     assert predictions is not None, "Model selection filter did not return predictions."
    #     assert isinstance(predictions, list), "Predictions should be a list."

    def test_maskrcnn_apply(self, test_image_dict, maskrcnn_model : MaskRCNNSegmentationFilter):
        # Assuming the model has a method called 'predict' that takes an image and returns predictions
        print(torch.version.cuda)
        print(torch.cuda.is_available())
        data = maskrcnn_model.apply(test_image_dict)
        assert data['segmentation_data'] is not None, "Model selection filter did not return predictions."
        assert isinstance(data['segmentation_data'], list), "Predictions should be a list."
    
    def test_load_model(self):
        pass