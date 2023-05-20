import numpy as np
from sklearn.metrics import pairwise_distances

class t_SNE:
    def __init__(self, n_components=2, perplexity=30, learning_rate=0.05, n_iter=1000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        
    def euclidean_distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)
    
    def calculate_perplexity(self, distances, perplexity):
        n = distances.shape[0]
        p_values = np.zeros((n, n))
        sigmas = np.ones(n)
        target_entropy = np.log2(perplexity)

        for i in range(n):
            beta_min = -np.inf
            beta_max = np.inf

            while True:
                distances_i = distances[i, np.concatenate((np.arange(0, i), np.arange(i + 1, n)))]
                similarities_i = np.exp(-distances_i * sigmas[i])
                entropy = -np.sum(similarities_i * np.log2(similarities_i))

                entropy_diff = entropy - target_entropy
                if np.abs(entropy_diff) <= 1e-7:
                    break

                if entropy_diff > 0:
                    beta_min = sigmas[i]
                    if beta_max == np.inf:
                        sigmas[i] *= 2
                    else:
                        sigmas[i] = (sigmas[i] + beta_max) / 2
                else:
                    beta_max = sigmas[i]
                    if beta_min == -np.inf:
                        sigmas[i] /= 2
                    else:
                        sigmas[i] = (sigmas[i] + beta_min) / 2

            p_values[i, np.concatenate((np.arange(0, i), np.arange(i + 1, n)))] = similarities_i / np.sum(similarities_i)

        return p_values

    def fit_transform(self, X):
        n = X.shape[0]
        distances = pairwise_distances(X)
        P = self.calculate_perplexity(distances, self.perplexity)

        Y = np.random.normal(0, 1e-4, (n, self.n_components))

        for i in range(self.n_iter):
            q_values = pairwise_distances(Y)
            q_values = 1 / (1 + q_values)
            np.fill_diagonal(q_values, 0)
            q_values /= np.sum(q_values)

            grad = np.zeros((n, self.n_components))
            for j in range(n):
                grad_j = 4 * np.dot((P[j] - q_values[j]) * q_values[j], Y[j] - Y)
                grad[j] = np.sum(grad_j, axis=0)

            Y -= self.learning_rate * grad

        return Y
  
