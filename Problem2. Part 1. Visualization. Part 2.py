import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# Load the data
X = np.load("C:/Users/lggr1/OneDrive/Escritorio/Data Analysis/Module 2/data/p2_unsupervised/X.npy")

# Log transformation
X_transformed = np.log2(X + 1)

# Standardize the data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X_transformed)

# Apply PCA
pca = PCA(n_components=0.85)
X_pca = pca.fit_transform(X_standardized)
print("Number of components that explain at least 85% of the variance:", X_pca.shape[1])

# Perform hierarchical clustering
Z = linkage(X_pca, method='ward')

# Cut the dendrogram to obtain the 3 main clusters
main_clusters = fcluster(Z, 3, criterion='maxclust')

# For each main cluster, perform K-Means clustering to find subclusters and calculate silhouette scores
subcluster_labels = np.zeros_like(main_clusters)
subcluster_count = {}
optimal_subclusters = {}

for cluster in np.unique(main_clusters):
    # Get the data points in the current main cluster
    cluster_indices = np.where(main_clusters == cluster)[0]
    cluster_data = X_pca[cluster_indices]

    # Determine the optimal number of subclusters using silhouette scores
    max_clusters = 11  # Adjust based on your data
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(cluster_data)
        silhouette_avg = silhouette_score(cluster_data, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    optimal_n_clusters = np.argmax(silhouette_scores) + 2
    optimal_subclusters[cluster] = optimal_n_clusters

    # Apply K-Means with the optimal number of subclusters
    kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
    subclusters = kmeans.fit_predict(cluster_data)

    # Store the subcluster labels
    subcluster_labels[cluster_indices] = subclusters + subcluster_count.get(cluster, 0)

    # Update subcluster count for the main cluster
    subcluster_count[cluster] = np.max(subclusters) + 1

    # Plot silhouette scores
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(cluster_data) + (optimal_n_clusters + 1) * 10])

    sample_silhouette_values = silhouette_samples(cluster_data, subclusters)
    y_lower = 10
    for i in range(optimal_n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[subclusters == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / optimal_n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_title("The silhouette plot for the various subclusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Subcluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual subclusters formed
    colors = cm.nipy_spectral(subclusters.astype(float) / optimal_n_clusters)
    ax2.scatter(cluster_data[:, 0], cluster_data[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")

    # Labeling the clusters
    centers = kmeans.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], marker="o", c="white", alpha=1, s=200, edgecolor="k")
    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st principal component")
    ax2.set_ylabel("Feature space for the 2nd principal component")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on main cluster {} with n_clusters = {}".format(cluster, optimal_n_clusters),
        fontsize=14, fontweight="bold")

    plt.show()

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_pca)

# Visualize the subclusters with t-SNE
plt.figure(figsize=(10, 7))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=subcluster_labels, cmap='viridis', s=10)
plt.title('t-SNE Visualization Colored by Subclusters')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.colorbar(label='Subcluster')
plt.show()

# Print the number of subclusters within each main cluster
print("Number of subclusters within each main cluster:")
for cluster, count in subcluster_count.items():
    print(f"Cluster {cluster}: {count} subclusters (Optimal: {optimal_subclusters[cluster]})")

