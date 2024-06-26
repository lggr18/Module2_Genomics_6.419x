import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load dataset
X = np.load("C:/Users/lggr1/OneDrive/Escritorio/Data Analysis/Module 2/data/p1/X.npy")

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to apply T-SNE and plot results
def tsne_perplexity(X, n_pcs, perplexity):
    pca = PCA(n_components=n_pcs)
    data_reduced = pca.fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    data_tsne = tsne.fit_transform(data_reduced)

    plt.figure(figsize=(8, 6))
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1])
    plt.title(f'T-SNE with {n_pcs} PCs, Perplexity={perplexity}')
    plt.show()


# Apply T-SNE with varying perplexity
for perplexity in [5, 10, 20, 30, 40, 50, 100]:
    tsne_perplexity(X_scaled, n_pcs=50, perplexity=perplexity)