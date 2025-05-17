import os
import tempfile
import sys

import torch
import pytest
import cv2

from app.flaskr import create_app

from app.filters.model_filter_factory import ModelFilterFactory
from app.pipeline.pipeline_builder import PipelineCreator
# from flaskr.db import get_db, init_db

# with open(os.path.join(os.path.dirname(__file__), 'data.sql'), 'rb') as f:
#     _data_sql = f.read().decode('utf8')


@pytest.fixture
def app():
    # db_fd, db_path = tempfile.mkstemp()

    app = create_app({
        'TESTING': True #,
        # 'DATABASE': db_path,
    })

    # with app.app_context():
    #     init_db()
    #     get_db().executescript(_data_sql)

    yield app

    # os.close(db_fd)
    # os.unlink(db_path)


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def runner(app):
    return app.test_cli_runner()

@pytest.fixture
def test_image():
    image = cv2.imread("tests/test_image.jpg") # TODO: replace with the actual path to test image
    if image is None:
        pytest.fail("Test image not found.")
    return image

@pytest.fixture
def test_yolo_image():
    image = cv2.imread("tests/page_101.png") # TODO: replace with the actual path to test image
    if image is None:
        pytest.fail("Test image not found.")
    return image

@pytest.fixture
def test_image_dict(test_image):
    # Convert the image to a dictionary format
    return {'image': test_image}

@pytest.fixture
def test_yolo_image_dict(test_yolo_image):
    # Convert the image to a dictionary format
    return {'image': test_yolo_image}

# @pytest.fixture
# def yolo_model():
#     # Load a pre-trained YOLO model for testing
#     return torch.load(pretrained=True)

@pytest.fixture
def maskrcnn_model(app):
    # Load a pre-trained Mask R-CNN model for testing
    factory = ModelFilterFactory(app.logger, app.config['RESOURCES_PATH'])
    return factory.create_model("maskRCNN")

@pytest.fixture
def graffitti_pipeline(app):
    pipelineCreator = PipelineCreator(app.logger, app.config['RESOURCES_PATH'])
    pipeline = pipelineCreator.construct_graffiti("maskRCNN")
    return pipeline