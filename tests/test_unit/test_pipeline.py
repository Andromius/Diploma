from pipeline.pipeline import Pipeline

def test_graffitti_pipeline(test_image, graffitti_pipeline : Pipeline):
    data = graffitti_pipeline.execute(test_image)
    print(data)
    assert data["image"] is not None, "Data structure doesnt contain image"
    assert isinstance(data, dict), "Data structure should be dictionary"
