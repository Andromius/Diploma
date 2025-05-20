from filters.filter import Filter
from logging import Logger

class SegmentThresholdFilter(Filter):
    def __init__(self, logger: Logger, threshold: int):
        super().__init__(name=__name__, logger=logger)
        self.threshold = threshold

    def apply(self, data: dict):
        if 'segmentation_data' not in data:
            raise ValueError("No image found in data.")
        
        segmentation_data = data['segmentation_data'][0]

        scores = segmentation_data['scores']
        boxes = segmentation_data['boxes']
        labels = segmentation_data['labels']
        masks = segmentation_data['masks']

        keep_mask = scores >= self.threshold

        filtered_scores = scores[keep_mask]
        filtered_boxes = boxes[keep_mask]
        filtered_labels = labels[keep_mask]
        filtered_masks = masks[keep_mask]

        data['segmentation_data'] = [{
            'scores': filtered_scores,
            'boxes': filtered_boxes,
            'labels': filtered_labels,
            'masks': filtered_masks
        }]

        return data


