import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Load your original data
X = np.load("C:/Users/lggr1/OneDrive/Escritorio/Data Analysis/Module 2/data/p2_unsupervised/X.npy")

# Step 2: Log transformation of the original data
log_data = np.log2(X + 1)

# Step 1: Clustering the log-transformed data into 11 clusters
optimal_n_clusters = 11
kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=10)
cluster_labels = kmeans.fit_predict(log_data)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(log_data, cluster_labels, test_size=0.2, random_state=10, stratify=cluster_labels)

# Step 3: Fit a logistic regression model using cluster labels as target
# Manually tune the regularization parameter using GridSearchCV
param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear'], 'max_iter': [10000]}
grid_search = GridSearchCV(LogisticRegression(random_state=10), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters from GridSearchCV
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train the final logistic regression model with the best parameters
log_reg = LogisticRegression(**best_params)
log_reg.fit(X_train, y_train)

# Evaluate the model
y_pred = log_reg.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
report = classification_report(y_val, y_pred)

print("Validation Accuracy:", accuracy)
print("Classification Report:\n", report)

# Save the model performance metrics including regularization parameter choice and validation performance
with open('C:/Users/lggr1/OneDrive/Escritorio/Data Analysis/Module 2/output/log_reg_performance.txt', 'w') as f:
    f.write(f"Best Parameters: {best_params}\n")
    f.write(f"Validation Accuracy: {accuracy}\n")
    f.write("Classification Report:\n")
    f.write(report)

# Step 4: Feature selection using logistic regression coefficients
coefficients = np.abs(log_reg.coef_).sum(axis=0)

# Select the top 100 features based on the coefficients
top_100_features_indices = np.argsort(coefficients)[-100:]

# Extract the top 100 features from the original log-transformed data
top_100_features = log_data[:, top_100_features_indices]

# Save the top 100 features
top_100_features_df = pd.DataFrame(top_100_features)
top_100_features_df.to_csv('C:/Users/lggr1/OneDrive/Escritorio/Data Analysis/Module 2/output/top_100_features.csv', index=False)

# Save the selected feature indices
selected_features_df = pd.DataFrame({'Selected Features': top_100_features_indices})
selected_features_df.to_csv('C:/Users/lggr1/OneDrive/Escritorio/Data Analysis/Module 2/output/selected_features.csv', index=False)
