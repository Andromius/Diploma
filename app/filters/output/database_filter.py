from filters.filter import Filter
from flaskr import db
from logging import Logger
import psycopg2
from flaskr.db import insert_graffiti_feature, get_all_graffiti_features

class DatabaseFilter(Filter):
    def __init__(self, logger: Logger, conn: psycopg2.extensions.connection):
        super().__init__(name="DatabaseFilter", logger=logger)
        self.conn = conn

    def apply(self, data: dict):
        if 'image' not in data:
            self.logger.error("No image found in data.")
            raise ValueError("No image found in data.")

        image = data['image']
        self.logger.info("Saving image to database.")

        colorfulness_data = data['colorfulness_data']
        dominant_colors = data['dominant_colors']
        warmth_scores = data['warmth_scores']
        global_color_histograms = data['global_color_histograms']
        mask_features = data['mask_features']

        for i in range(len(colorfulness_data)):
            vector = []
            vector.append(warmth_scores[i])
            vector.append(colorfulness_data[i])
            vector.extend(dominant_colors[i])
            vector.extend(global_color_histograms[i])
            vector.extend(mask_features[i])
            insert_graffiti_feature(
                conn=self.conn,
                image_name=data['image_name'],
                segment_id=i,
                feature_vector=vector,
                mask=data['segmentation_data'][0]['masks'].cpu().numpy()[i],
                logger=self.logger
            )
            self.logger.info(f"Feature vector {data['image_name']}:{i} saved successfully.")


        data['feat_vecs'] = get_all_graffiti_features(self.conn, self.logger)

        return data