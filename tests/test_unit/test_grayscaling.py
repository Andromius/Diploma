from app.filters.preprocessing.grayscaling import GrayscaleFilter

def test_grayscaling_conversion(test_image_dict,app):
    filter = GrayscaleFilter(app.logger)
    data = filter.apply(test_image_dict)
    assert len(data['grayscale'].shape) == 2, "Image is not grayscaled."
