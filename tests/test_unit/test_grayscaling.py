from filters.preprocessing.grayscaling import GrayscaleFilter

def test_grayscaling_conversion(test_image,app):
    filter = GrayscaleFilter(app.logger)
    gray_image = filter.apply(test_image)
    assert len(gray_image.shape) == 2, "Image is not grayscaled."
