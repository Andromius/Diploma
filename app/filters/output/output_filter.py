from filters.filter import Filter
from logging import Logger
import numpy as np
import psycopg2
from flaskr import db
import pickle
import cv2

class OutputFilter(Filter):
    def __init__(self, logger : Logger, conn: psycopg2.extensions.connection):
        super().__init__(__name__, logger)
        self.conn = conn

    def apply(self, data : dict):

        self.logger.info("Applying output filter")
        prediction = data["segmentation_data"]
        image = data["image"]
        masks = prediction[0]['masks'].cpu().numpy()  # Get predicted masks
        hdbscan_labels = data.get("hdbscan_labels", None)
        images = []
        rows = data['feat_vecs']
        final_similar_images = []
        for i, mask in enumerate(masks):
            binary_mask = mask[0, :, :] > 0.8
            self.logger.info(f"Processing mask {i+1}/{len(masks)}, clustster_id: {hdbscan_labels[i-len(masks)]}")
            similar_indices = np.where(hdbscan_labels == hdbscan_labels[i-len(masks)])[0]
            self.logger.info(f"Found {len(similar_indices)} similar images for mask {i+1}")
            sim_images = []
            for j in similar_indices:
                self.logger.info(f"Processing similar image {j}/{len(rows)}")
                sim_image = cv2.imread(f'uploads/{rows[j][0]}')
                sim_binary_mask = pickle.loads(rows[j][3])
                sim_binary_mask = sim_binary_mask[0, :, :] > 0.8
                
                y_indices, x_indices = np.where(sim_binary_mask)
                if len(x_indices) > 0 and len(y_indices) > 0:
                    # Get shape of the mask
                    shape_mask = np.zeros_like(sim_binary_mask, dtype=bool)
                    shape_mask[y_indices, x_indices] = True
                    
                    # Extract just those pixels from the original image
                    # We need to determine what pixels to include
                    unique_y = sorted(set(y_indices))
                    unique_x = sorted(set(x_indices))
                    
                    # Create a mapping from original coordinates to our new compact image
                    y_map = {y: i for i, y in enumerate(unique_y)}
                    x_map = {x: i for i, x in enumerate(unique_x)}
                    
                    # Create a new image with just the required size
                    new_h, new_w = len(unique_y), len(unique_x)
                    if len(sim_image.shape) == 3 and sim_image.shape[2] == 3:  # RGB
                        cutout = np.zeros((new_h, new_w, 4), dtype=np.uint8)
                        
                        # Place only the masked pixels into the new image
                        for orig_y, orig_x in zip(y_indices, x_indices):
                            new_y, new_x = y_map[orig_y], x_map[orig_x]
                            cutout[new_y, new_x, :3] = sim_image[orig_y, orig_x]
                            cutout[new_y, new_x, 3] = 255
                    else:  # Grayscale
                        cutout = np.zeros((new_h, new_w, 4), dtype=np.uint8)
                        for orig_y, orig_x in zip(y_indices, x_indices):
                            new_y, new_x = y_map[orig_y], x_map[orig_x]
                            cutout[new_y, new_x, :3] = sim_image[orig_y, orig_x]
                            cutout[new_y, new_x, 3] = 0
                    
                    sim_images.append(cutout)
            final_similar_images.append(sim_images)
            # Get the bounding box of the mask
            y_indices, x_indices = np.where(binary_mask)
            if len(x_indices) > 0 and len(y_indices) > 0:
                # Get shape of the mask
                shape_mask = np.zeros_like(binary_mask, dtype=bool)
                shape_mask[y_indices, x_indices] = True
                
                # Extract just those pixels from the original image
                # We need to determine what pixels to include
                unique_y = sorted(set(y_indices))
                unique_x = sorted(set(x_indices))
                
                # Create a mapping from original coordinates to our new compact image
                y_map = {y: i for i, y in enumerate(unique_y)}
                x_map = {x: i for i, x in enumerate(unique_x)}
                
                # Create a new image with just the required size
                new_h, new_w = len(unique_y), len(unique_x)
                if len(image.shape) == 3 and image.shape[2] == 3:  # RGB
                    cutout = np.zeros((new_h, new_w, 4), dtype=np.uint8)
                    
                    # Place only the masked pixels into the new image
                    for orig_y, orig_x in zip(y_indices, x_indices):
                        new_y, new_x = y_map[orig_y], x_map[orig_x]
                        cutout[new_y, new_x, :3] = image[orig_y, orig_x]
                        cutout[new_y, new_x, 3] = 255
                else:  # Grayscale
                    cutout = np.zeros((new_h, new_w, 4), dtype=np.uint8)
                    for orig_y, orig_x in zip(y_indices, x_indices):
                        new_y, new_x = y_map[orig_y], x_map[orig_x]
                        cutout[new_y, new_x, :3] = image[orig_y, orig_x]
                        cutout[new_y, new_x, 3] = 0
                
                images.append(cutout)

        data["final_images"] = images
        data["similar_images"] = final_similar_images
        return data