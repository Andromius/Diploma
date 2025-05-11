from pipeline.pipeline import Pipeline

def test_graffitti_pipeline(test_image, graffitti_pipeline : Pipeline):
    data = graffitti_pipeline.execute(test_image)
    assert data["image"] is not None, "Data structure doesnt contain image"
    assert isinstance(data, dict), "Data structure should be dictionary"

    assert "final_images" in data, "Output does not contain 'final_images' key."
    
    assert len(data["final_images"]) > 0, "No final images generated."
    
    for img in data["final_images"]:
        assert len(img.shape) == 3, "Final image is not in the expected format."
        assert img.shape[2] == 4, "Final image does not have 4 channels (RGBA)."
