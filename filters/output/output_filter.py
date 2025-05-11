from filters.filter import Filter
from logging import Logger
from matplotlib import pyplot as plt
import numpy as np

class OutputFilter(Filter):
    def __init__(self, logger : Logger, threshold : float = 0.8, name : str = "output"):
        super().__init__(name, logger)
        self.threshold = threshold

    def apply(self, data : dict):
        prediction = data["segmentation_data"]
        image = data["image"]
        masks = prediction[0]['masks'].cpu().numpy()  # Get predicted masks
        labels = prediction[0]['labels'].cpu().numpy()  # Get predicted labels
        scores = prediction[0]['scores'].cpu().numpy()  # Get predicted scores
        print(prediction[0].keys())

        # Set up the figure size to control the image size
        fig_size = (10, 10)
        fig = plt.figure(figsize=fig_size)  # Adjust the figure size here
        plt.imshow(image)  # Display the image
        ax = plt.gca()

        arr = list(zip(masks, labels, scores))

        images = []

        for i, (mask, label, score) in enumerate(arr):
            if score > self.threshold:
                # Convert mask to a boolean array and squeeze the channel dimension
                mask = mask[0, :, :] > 0.5
                color = np.random.rand(3)  # Generate a random color for each mask
                # Create an RGBA array for the mask with the random color and alpha
                mask_rgba = np.concatenate([np.tile(color, (mask.shape[0], mask.shape[1], 1)), mask[:, :, np.newaxis] * 0.7], axis=-1)
                ax.imshow(mask_rgba, extent=(0, image.shape[1], image.shape[0], 0), interpolation='nearest')
                class_name = "Graffiti"  # Get the class name
                # Get the bounding box of the mask to place the label
                y, x = np.where(mask)
                if len(x) > 0 and len(y) > 0:
                    x_min, x_max = np.min(x), np.max(x)
                    y_min, y_max = np.min(y), np.max(y)
                    ax.text(x_min, y_min, f"{class_name} ({score:.2f})", color='w', fontsize=10, backgroundcolor='black')

        images.append(image)

        data["final_images"] = images
        return data