import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import seaborn as sns

# Load the dataset
X = np.load("C:/Users/lggr1/OneDrive/Escritorio/Data Analysis/Module 2/data/p2_unsupervised/X.npy")

# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction
n_pcs = 50
pca = PCA(n_components=n_pcs)
X_pca = pca.fit_transform(X_scaled)


# Function to apply T-SNE and plot results
def tsne_clustering(data, n_pcs, perplexity, learning_rate, method):
    pca = PCA(n_components=n_pcs)
    data_reduced = pca.fit_transform(data)
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
    data_tsne = tsne.fit_transform(data_reduced)

    # Perform hierarchical clustering
    Z = linkage(data_tsne, method=method)
    clusters = fcluster(Z, t=5, criterion='maxclust')

    # Plot the T-SNE and clustering results
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data_tsne[:, 0], y=data_tsne[:, 1], hue=clusters, palette='viridis', s=50)
    plt.title(f'T-SNE with Perplexity={perplexity}, Learning Rate={learning_rate}, Clustering={method.capitalize()}')
    plt.show()


# Apply T-SNE and hierarchical clustering with various combinations
perplexities = [5, 30, 50]
learning_rates = [10, 200, 1000]
methods = ['single', 'ward']

for perplexity in perplexities:
    for learning_rate in learning_rates:
        for method in methods:
            tsne_clustering(X_scaled, n_pcs=50, perplexity=perplexity, learning_rate=learning_rate, method=method)
