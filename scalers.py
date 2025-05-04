from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import numpy as np

class ClusterBasedScaler(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.clusterer = GaussianMixture(n_components=n_clusters)
        self.scalers = {}

    def fit(self, X, y=None):
        self.cluster_labels_ = self.clusterer.fit_predict(X)
        for cluster in range(self.n_clusters):
            scaler = StandardScaler()
            cluster_data = X[self.cluster_labels_ == cluster]
            scaler.fit(cluster_data)
            self.scalers[cluster] = scaler
        return self

    def transform(self, X):
        labels = self.clusterer.predict(X)
        X_scaled = np.zeros_like(X)
        for cluster in range(self.n_clusters):
            cluster_indices = (labels == cluster)
            X_scaled[cluster_indices] = self.scalers[cluster].transform(X[cluster_indices])
        return X_scaled