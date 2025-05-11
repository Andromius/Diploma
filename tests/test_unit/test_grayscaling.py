from filters.preprocessing.grayscaling import GrayscaleFilter

def test_grayscaling_conversion(test_image,app):
    filter = GrayscaleFilter(app.logger)
    data = filter.apply(test_image)
    assert len(data['grayscale'].shape) == 2, "Image is not grayscaled."
