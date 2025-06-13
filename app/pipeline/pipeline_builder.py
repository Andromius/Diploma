from filters.preprocessing import contouring, gaussianblur, grayscaling, histogram_equalization, morphology, edge_separation, thresholding
from filters.output import output_filter
from filters.analysis import similarity_filter
from filters.model_filter_factory import ModelFilterFactory
from pipeline.pipeline import Pipeline
from logging import Logger
import psycopg2

class PipelineBuilder:
    def __init__(self, logger, resources_path):
        self.resources_path = resources_path
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
    
    def extract_feature_vector(self):
        self.pipeline.add_filter(ModelFilterFactory(self.logger).create_model("feature_vector_extractor"))
        return self
    
    def calculate_similarity(self, name, threshold=0.5):
        self.pipeline.add_filter(similarity_filter.SimilarityFilter(self.logger, name, threshold))
        return self
    
    def separate_lines(self):
        self.pipeline.add_filter(edge_separation.LineSeparationFilter(self.logger))
        return self
    
    def morphological_operations(self, kernel_size=(3, 3), iterations=5):
        self.pipeline.add_filter(morphology.MorphologicalOperationsFilter(self.logger, kernel_size, iterations))
        return self
    
    def thresholding(self, threshold=0.8):
        self.pipeline.add_filter(thresholding.ThresholdingFilter(self.logger, threshold))
        return self

    def build(self):
        return self.pipeline

class PipelineCreator:
    def __init__(self, logger : Logger, resources_path: str, connection: psycopg2.extensions.connection):
        self.builder = PipelineBuilder(logger, resources_path)
        self.logger = logger
        self.connection = connection
        
    def construct_voynich(self, model_type : str):
        return self.builder.segmentation_model(model_type).build()
    
    def construct_graffiti(self, model_type : str):
        return self.builder.segmentation_model(model_type)\
                            .gaussian_blur()\
                            .thresholding()\
                            .morphological_operations()\
                            .output()\
                            .separate_lines()\
                            .extract_feature_vector()\
                            .build()