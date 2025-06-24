import hdbscan as hdb
import numpy as np
import flaskr.db as db
from filters.filter import Filter
from logging import Logger
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import StandardScaler
import os

class HDBSCANFilter(Filter):
    def __init__(self, min_samples=1, min_cluster_size=2, logger=None, conn=None):
        super().__init__(name=__name__, logger=logger)
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.clusterer = hdb.HDBSCAN(min_cluster_size=self.min_cluster_size, 
                            min_samples=self.min_samples,
                            allow_single_cluster=True,
                            cluster_selection_epsilon=0.01, 
                            metric='precomputed',
                            core_dist_n_jobs=2)
        self.conn = conn

    def apply(self, data: dict):
        if 'fetched' not in data:
            self.logger.error("No feature vectors found in data.")
            raise ValueError("No feature vectors found in data.")

        if len(data["fetched"]) == 0:
            self.logger.warning("No feature vectors to cluster.")
            return data

        if len(data["fetched"]) < self.min_cluster_size:
            self.logger.warning(f"Not enough feature vectors ({len(data['fetched'])}) to form a cluster of size {self.min_cluster_size}")
            return data
        
        for vec in data['fetched']:
            self.logger.info(f"Fetched row: {vec}")
        
        cleaned_feat_vecs = [vec[2] for vec in data['fetched']]
        X_scaled = StandardScaler().fit_transform(cleaned_feat_vecs)
        distance_matrix = cosine_distances(cleaned_feat_vecs)
        self.logger.info(f"Mean: {X_scaled.mean(axis=0)}, Variance: {X_scaled.var(axis=0)}, Min: {X_scaled.min(axis=0)}, Max: {X_scaled.max(axis=0)}")
        
        self.logger.info("Applying HDBSCAN clustering.")
        self.logger.info(f"Distance matrix sample: {distance_matrix[:5, :5]}")

        labels = self.clusterer.fit(distance_matrix).labels_
        probabilities = self.clusterer.probabilities_
        self.logger.info(f"Cluster labels: {labels}")
        self.logger.info(f"Cluster probabilities: {probabilities}")

        data['hdbscan_labels'] = labels
        data['hdbscan_probs'] = probabilities

        self.logger.info(f"Clustering completed with {len(set(labels))} clusters found.")
        return data