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

        self.logger.info("Saving image to database.")

        self.logger.info(f"Final features length: {len(data['final_features'])}")

        for i in range(len(data['final_features'])):
            vector = data['final_features'][i]

            insert_graffiti_feature(
                conn=self.conn,
                image_name=data['image_name'],
                segment_id=i,
                feature_vector=vector,
                mask=data['segmentation_data'][0]['masks'].cpu().numpy()[i],
                logger=self.logger
            )
            self.logger.info(f"Feature vector {data['image_name']}:{i} saved successfully.")

        data["fetched"] = get_all_graffiti_features(self.conn, self.logger)
        self.logger.info(f"Sample fetched row: {data['fetched'][0]}")
        return data