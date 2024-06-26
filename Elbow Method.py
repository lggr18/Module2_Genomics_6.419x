import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


file_path = "C:/Users/lggr1/OneDrive/Escritorio/Data Analysis/Module 2/data/p1/X.npy"

X = np.load(file_path)

# Apply log2(x + 1)
X_transformed = np.log2(X + 1)

pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_transformed)

# Elbow method to determine the optimal number of clusters
inertia = []
K = range(1, 15)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WGSS)')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()

# Select the optimal number of clusters
optimal_clusters = 5

# WGSS
kmeans_optimal = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans_optimal.fit(X_pca)
wgss_optimal = kmeans_optimal.inertia_

print(f"Number of clusters: {optimal_clusters}")
print(f"WGSS = {wgss_optimal:.2f}")