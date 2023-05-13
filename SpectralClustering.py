import numpy as np
from sklearn.cluster import KMeans


class SpectralClustering:
    def __init__(self, k=2):
        self.k = k
        self.kmeans = KMeans(n_clusters=self.k)

    def fit(self, X):
        pairwise_dist = np.sqrt(((X[:, np.newaxis, :] - X) ** 2).sum(axis=2))

        median_dist = np.median(pairwise_dist)
        gamma = 1 / (2 * median_dist ** 2)
        W = np.exp(-gamma * pairwise_dist ** 2)

        D = np.diag(np.sum(W, axis=1))
        L = D - W      
         
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        indices = np.argsort(eigenvalues)[:self.k]
        X_new = eigenvectors[:, indices]

        self.kmeans.fit(X_new)

        self.labels_ = self.kmeans.labels_
    def predict(self, X):
        pairwise_dist = np.sqrt(((X[:, np.newaxis, :] - X) ** 2).sum(axis=2))

        median_dist = np.median(pairwise_dist)
        gamma = 1 / (2 * median_dist ** 2)
        W = np.exp(-gamma * pairwise_dist ** 2)

        D = np.diag(np.sum(W, axis=1))

        L = D - W

        eigenvalues, eigenvectors = np.linalg.eigh(L)

        indices = np.argsort(eigenvalues)[:self.k]
        X_new = eigenvectors[:, indices]
        return self.kmeans.predict(eigenvectors)
