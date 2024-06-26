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
def tsne_learning_rate(data, n_pcs, learning_rate):
    pca = PCA(n_components=n_pcs)
    data_reduced = pca.fit_transform(data)
    tsne = TSNE(n_components=2, learning_rate=learning_rate, random_state=42)
    data_tsne = tsne.fit_transform(data_reduced)

    plt.figure(figsize=(8, 6))
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1])
    plt.title(f'T-SNE with {n_pcs} PCs, Learning Rate={learning_rate}')
    plt.show()


# Apply T-SNE with varying learning rates
for learning_rate in [10, 50, 100, 200, 500, 1000]:
    tsne_learning_rate(X_scaled, n_pcs=50, learning_rate=learning_rate)
