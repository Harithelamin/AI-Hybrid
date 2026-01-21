# Unsupervised Learning (K-Means)
# the goal is to group data without labels.

import numpy as np
from sklearn.cluster import KMeans

# Data points
X = np.array([
    [1, 2], [1, 4], [1, 0],
    [10, 2], [10, 4], [10, 0]
])

# Train KMeans
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)

print("Cluster centers:\n", kmeans.cluster_centers_)
print("Labels:", kmeans.labels_)
