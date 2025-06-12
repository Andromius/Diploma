from filters.filter import Filter
from sklearn.cluster import KMeans
import numpy as np
from logging import Logger

class DominantColorsFilter(Filter):
    def __init__(self, logger: Logger, n_colors: int = 5):
        super().__init__(name=__name__, logger=logger)
        self.n_colors = n_colors

    def apply(self, data: dict):
        if 'image' not in data:
            raise ValueError("No image found in data.")
        
        prediction = data["segmentation_data"][0]
        image = data["image"]
        masks = prediction['masks'].cpu().numpy()  # Get predicted masks
        
        dominant_colors = []
        
        for mask in masks:
            binary_mask = mask[0, :, :] > 0.8

            masked_img = image * binary_mask[:, :, np.newaxis]
            non_black_mask = ~np.all(masked_img == [0, 0, 0], axis=2)
            masked_img = masked_img[non_black_mask]

            kmeans = KMeans(n_clusters=self.n_colors, random_state=0)
            kmeans.fit(masked_img)
            proportions = np.bincount(kmeans.labels_) / len(kmeans.labels_)
            indices = np.argsort(proportions)[::-1]
            colors = kmeans.cluster_centers_.astype(int)
            sorted_colors = []
            for index in indices:
                sorted_colors.append(colors[index][0] / 255)
                sorted_colors.append(colors[index][1] / 255)
                sorted_colors.append(colors[index][2] / 255)
                sorted_colors.append(proportions[index])

            dominant_colors.append(sorted_colors)


        data['dominant_colors'] = dominant_colors
        return data
