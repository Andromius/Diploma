from filters.preprocessing import contouring, gaussianblur, grayscaling, histogram_equalization
from filters.output import output_filter
from filters.model_filter_factory import ModelFilterFactory
from pipeline.pipeline import Pipeline
from logging import Logger

class PipelineBuilder:
    def __init__(self, logger):
        self.pipeline = Pipeline()
        self.logger = logger

    def contouring(self):
        self.pipeline.add_filter(contouring.ContouringFilter(self.logger))
        return self

    def gaussian_blur(self):
        self.pipeline.add_filter(gaussianblur.GaussianBlurFilter(self.logger))
        return self

    def grayscaling(self):
        self.pipeline.add_filter(grayscaling.GrayscaleFilter(self.logger))
        return self

    def histogram_equalization(self):
        self.pipeline.add_filter(histogram_equalization.HistogramEqualizationFilter(self.logger))
        return self
    
    def segmentation_model(self, name):
        model_factory = ModelFilterFactory(self.logger)
        self.pipeline.add_filter(model_factory.create_model(name))
        return self
    
    def output(self):
        self.pipeline.add_filter(output_filter.OutputFilter(self.logger))
        return self

    def build(self):
        return self.pipeline

class PipelineCreator:
    def __init__(self, logger : Logger):
        self.builder = PipelineBuilder(logger)
        self.logger = logger
        
    def construct_voynich(self, model_type : str):
        return self.builder.segmentation_model(model_type).build()
    
    def construct_graffiti(self, model_type : str):
        return self.builder.segmentation_model(model_type).output().build()