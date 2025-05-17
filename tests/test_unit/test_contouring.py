from app.filters.preprocessing.contouring import ContouringFilter  


def test_contours_detection(test_image_dict,app):
    contour_filter = ContouringFilter(app.logger)
    output = contour_filter.apply(test_image_dict)
    assert len(output) > 0, "No contours detected."