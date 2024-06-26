import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load your dataset
X = np.load("C:/Users/lggr1/OneDrive/Escritorio/Data Analysis/Module 2/data/p1/X.npy")

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
pca.fit(data_scaled)

# Calculate cumulative explained variance ratio
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot cumulative explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance as a Function of the Number of Principal Components')
plt.grid(True)
plt.show()

# Print cumulative explained variance at specific points
print(f'Cumulative explained variance by 10 PCs: {cumulative_explained_variance[9]:.4f}')
print(f'Cumulative explained variance by 50 PCs: {cumulative_explained_variance[49]:.4f}')
print(f'Cumulative explained variance by 100 PCs: {cumulative_explained_variance[99]:.4f}')
print(f'Cumulative explained variance by 250 PCs: {cumulative_explained_variance[249]:.4f}')
print(f'Cumulative explained variance by 500 PCs: {cumulative_explained_variance[499]:.4f}')
