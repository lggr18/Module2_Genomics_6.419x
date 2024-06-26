import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Ruta del archivo en formato de cadena
file_path = "C:/Users/lggr1/OneDrive/Escritorio/Data Analysis/Module 2/data/p1/X.npy"

# Cargar el archivo .npy
X = np.load(file_path)

# Aplicar la transformación logarítmica log2(x + 1)
X_transformed = np.log2(X + 1)

# Proyectar los datos en los 50 principales componentes principales (PCs)
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_transformed)

# Método del codo para determinar el número óptimo de clusters
inertia = []
K = range(1, 15)  # Probar con un rango de 1 a 15 clusters

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(8, 6))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WGSS)')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()

# Seleccionar el número óptimo de clusters visualmente
optimal_clusters = 5  # Por ejemplo, si visualmente determinas que el codo está en K=5

# Obtener el valor de WGSS para el número óptimo de clusters
kmeans_optimal = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans_optimal.fit(X_pca)
wgss_optimal = kmeans_optimal.inertia_

print(f"Number of clusters: {optimal_clusters}")
print(f"WGSS = {wgss_optimal:.2f}")