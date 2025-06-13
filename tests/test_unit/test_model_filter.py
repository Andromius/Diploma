import pytest
import pandas as pd
from .test_filter import TestFilter
from filters.preprocessing.model_selection_filter import ModelSelectionFilter

class TestModelFilter(TestFilter):

    def test_yolo_apply(self, test_yolo_image, app):
        # Assuming the model has a method called 'predict' that takes an image and returns predictions
        filter = ModelSelectionFilter("yolo", app.logger)
        predictions = filter.apply(test_yolo_image)
        assert predictions is not None, "Model selection filter did not return predictions."
        assert isinstance(predictions, list), "Predictions should be a list."

    def test_maskrcnn_apply(self, test_image, app):
        # Assuming the model has a method called 'predict' that takes an image and returns predictions
        filter = ModelSelectionFilter("mask", app.logger)
        predictions = filter.apply(test_image)
        assert predictions is not None, "Model selection filter did not return predictions."
        assert isinstance(predictions, list), "Predictions should be a list."
    
    def test_load_model(self):
        pass