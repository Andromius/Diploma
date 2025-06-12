from filters.filter import Filter
import cv2
from logging import Logger
import numpy as np
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_distances

class HDBSCANFilter(Filter):
    def __init__(self, logger: Logger, min_cluster_size: int, min_samples: int):
        super().__init__(name="HDBSCANFilter", logger=logger)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def apply(self, data: dict):
        if 'feat_vecs' not in data:
            self.logger.error("No feature vectors found in data.")
            raise ValueError("No feature vectors found in data.")

        if len(data["feat_vecs"]) == 0:
            self.logger.warning("No feature vectors to cluster.")
            return data

        if len(data["feat_vecs"]) < self.min_cluster_size:
            self.logger.warning(f"Not enough feature vectors ({len(data['feat_vecs'])}) to form a cluster of size {self.min_cluster_size}")
            return data

        cleaned_feat_vecs = [vec[2] for vec in data['feat_vecs']]
        distance_matrix = cosine_distances(cleaned_feat_vecs)
        clusterer = HDBSCAN(min_cluster_size=self.min_cluster_size, 
                            min_samples=self.min_samples,
                            allow_single_cluster=True,
                            cluster_selection_epsilon=0.01, 
                            metric='precomputed',
                            core_dist_n_jobs=2)
        cluster_labels = clusterer.fit(distance_matrix).labels_
        cluster_probs = clusterer.probabilities_
        self.logger.info(f"Cluster labels: {cluster_labels}")
        self.logger.info(f"Cluster probabilities: {cluster_probs}")

        data['hdbscan_labels'] = cluster_labels
        data['hdbscan_probs'] = cluster_probs

        return data