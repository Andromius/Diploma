from filters.filter import Filter
from logging import Logger
from torch import nn
import torch
import torchvision.transforms.functional as F

class SegmentThresholdFilter(Filter):
    def __init__(self, logger: Logger, threshold: int):
        super().__init__(name=__name__, logger=logger)
        self.threshold = threshold

    def apply(self, data: dict):
        if 'segmentation_data' not in data:
            raise ValueError("No image found in data.")
        
        self.logger.info(f"Applying {self.name} with threshold {self.threshold}")
        segmentation_data = data['segmentation_data'][0]

        scores = segmentation_data['scores']
        boxes = segmentation_data['boxes']
        labels = segmentation_data['labels']
        masks = segmentation_data['masks']
        features = data['mask_features']

        keep_mask = scores >= self.threshold

        filtered_scores = scores[keep_mask]
        filtered_boxes = boxes[keep_mask]
        filtered_labels = labels[keep_mask]
        filtered_masks = masks[keep_mask]
        filtered_features = features[keep_mask]

        data['segmentation_data'] = [{
            'scores': filtered_scores,
            'boxes': filtered_boxes,
            'labels': filtered_labels,
            'masks': filtered_masks
        }]

        self.logger.info(f"Applying {self.name} with threshold {self.threshold}")
        data['mask_features'] = filtered_features

        if filtered_features.numel() > 0:
            self.logger.info(f"Actual shape: {filtered_features.shape}")
            mask_resolution_h, mask_resolution_w = filtered_features.shape[2:]

            binary_masks = (torch.sigmoid(filtered_masks) > 0.5).float()
            resized_binary_masks = F.resize(binary_masks, (mask_resolution_h, mask_resolution_w), interpolation=F.InterpolationMode.NEAREST)

            masked_features = filtered_features * resized_binary_masks

            self.logger.info(f"Shape of masked features: {masked_features.shape}")

            pooled_masked_features = torch.nn.functional.adaptive_avg_pool2d(masked_features, (1, 1)).squeeze(-1).squeeze(-1)
            self.logger.info(f"Shape of pooled masked features (per graffiti): {pooled_masked_features.shape}")
            data['mask_features'] = pooled_masked_features

        return data


