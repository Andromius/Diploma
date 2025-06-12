from filters.preprocessing import contouring, gaussianblur, grayscaling, histogram_equalization, segment_threshold_filter, morphological_operations_filter
from filters.output import output_filter, database_filter
from filters.analysis import colorfulness_filter, dominant_colors_filter, color_warmth_filter, global_color_histogram_filter
from filters.prediction import hdbscan_filter
from filters.model_filter_factory import ModelFilterFactory
from pipeline.pipeline import Pipeline
from logging import Logger
import psycopg2

class PipelineBuilder:
    def __init__(self, logger, resources_path):
        self.pipeline = Pipeline()
        self.logger = logger
        self.resources_path = resources_path

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
        model_factory = ModelFilterFactory(self.logger, self.resources_path)
        self.pipeline.add_filter(model_factory.create_model(name))
        return self
    
    def output(self, connection: psycopg2.extensions.connection):
        self.pipeline.add_filter(output_filter.OutputFilter(self.logger, connection))
        return self
    
    def segment_threshold(self, threshold):
        self.pipeline.add_filter(segment_threshold_filter.SegmentThresholdFilter(self.logger, threshold))
        return self
    
    def colorfulness(self):
        self.pipeline.add_filter(colorfulness_filter.ColorfulnessFilter(self.logger))
        return self
    
    def morphological_operations(self):
        self.pipeline.add_filter(morphological_operations_filter.MorphologicalOperationsFilter(self.logger))
        return self
    
    def dominant_colors(self):
        self.pipeline.add_filter(dominant_colors_filter.DominantColorsFilter(self.logger))
        return self
    
    def hdbscan(self, min_cluster_size: int = 2, min_samples: int = 1):
        self.pipeline.add_filter(hdbscan_filter.HDBSCANFilter(self.logger, min_cluster_size, min_samples))
        return self
    
    def color_warmth(self):
        self.pipeline.add_filter(color_warmth_filter.ColorWarmthFilter(self.logger))
        return self
    
    def global_color_histogram(self):
        self.pipeline.add_filter(global_color_histogram_filter.GlobalColorHistogramFilter(self.logger))
        return self
    
    def database(self, connection: psycopg2.extensions.connection):
        self.pipeline.add_filter(database_filter.DatabaseFilter(self.logger, connection))
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
    
    def construct_graffiti(self, model_type : str, threshold: int = 0.9):
        return self.builder.segmentation_model(model_type)\
                            .segment_threshold(threshold)\
                            .dominant_colors()\
                            .colorfulness()\
                            .color_warmth()\
                            .global_color_histogram()\
                            .database(self.connection)\
                            .hdbscan()\
                            .output(self.connection)\
                            .build()