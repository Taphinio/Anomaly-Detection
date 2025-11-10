import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


X_train, Y = make_blobs(n_samples=500, n_features=2, centers=[[0,0]], cluster_std=1.0, random_state=42)


projection_vectors = []
for Y in range(5):
    v = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]])
    projection_vectors.append(v / np.linalg.norm(v))
projection_vectors = np.array(projection_vectors)

X_test = np.random.uniform(-3, 3 , (500,2))


def compute_histograms(X, vectors, n_bins):
    histograms, bin_edges = [], []
    for v in vectors:
        projected = X @ v
        hist_range = (projected.min() - 1, projected.max() + 1)
        hist, edges = np.histogram(projected, bins=n_bins, range=hist_range)
        histograms.append(hist / len(X))
        bin_edges.append(edges)
    return histograms, bin_edges

def compute_scores(X_test, vectors, histograms, bin_edges):
    scores = []
    for x in X_test:
        probs = []
        for i in range(len(vectors)):  
            val = x @ vectors[i]       
            idx = np.digitize(val, bin_edges[i]) - 1
            idx = np.clip(idx, 0, len(histograms[i]) - 1)
            probs.append(histograms[i][idx])
        scores.append(np.mean(probs))
    return np.array(scores)

bins_to_test = [5, 10, 20, 50]
plt.figure(figsize=(15, 10))

for i in range(len(bins_to_test)):  
    n_bins = bins_to_test[i]
    histograms, bin_edges = compute_histograms(X_train, projection_vectors, n_bins)
    anomaly_scores = compute_scores(X_test, projection_vectors, histograms, bin_edges)
    
    plt.subplot(2, 2, i + 1)  
    scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=anomaly_scores, 
                         cmap='viridis', s=30, alpha=0.7)
    plt.colorbar(scatter, label='Score')
    plt.title(f'{n_bins} Bins')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)

plt.suptitle('Ex1: Effect of Number of Bins on Anomaly Scores', fontsize=14)
plt.tight_layout()
plt.savefig("Ex1.png")
plt.show()
