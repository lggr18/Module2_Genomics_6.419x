import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import seaborn as sns

# Load the dataset
X = np.load("C:/Users/lggr1/OneDrive/Escritorio/Data Analysis/Module 2/data/p2_unsupervised/X.npy")

# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)

# Hierarchical clustering using Single Linkage
Z_single = linkage(X_pca, method='single')
clusters_single = fcluster(Z_single, t=5, criterion='maxclust')

# Hierarchical clustering using Ward's method
Z_ward = linkage(X_pca, method='ward')
clusters_ward = fcluster(Z_ward, t=5, criterion='maxclust')

# Visualize the resulting clusters
def plot_clusters(X, clusters, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=clusters, palette='viridis', s=50)
    plt.title(title)
    plt.show()

# Plot clusters for Single Linkage
plot_clusters(X_pca, clusters_single, "Hierarchical Clustering with Single Linkage")

# Plot clusters for Ward's Method
plot_clusters(X_pca, clusters_ward, "Hierarchical Clustering with Ward's Method")
