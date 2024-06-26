import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
X = np.load("C:/Users/lggr1/OneDrive/Escritorio/Data Analysis/Module 2/data/p1/X.npy")

# Standardize the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_scaled = scaler.fit_transform(X)

# Define the number of PCs to test
num_pcs = [10, 50, 100, 250, 500]

# Dictionary to store T-SNE results
tsne_results = {}

for n_pc in num_pcs:
    # Apply PCA
    pca = PCA(n_components=n_pc)
    data_pca = pca.fit_transform(data_scaled)

    # Apply T-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000)
    tsne_result = tsne.fit_transform(data_pca)

    # Store the result
    tsne_results[n_pc] = tsne_result

    # Plotting the result
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], palette='viridis')
    plt.title(f'T-SNE with {n_pc} PCs')
    plt.show()


