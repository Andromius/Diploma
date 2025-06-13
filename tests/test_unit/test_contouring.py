from filters.preprocessing.contouring import ContouringFilter  


def test_contours_detection(test_image,app):
    contour_filter = ContouringFilter(app.logger)
    output = contour_filter.apply(test_image)
    assert len(output) > 0, "No contours detected."