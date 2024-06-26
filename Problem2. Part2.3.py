import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

# Load data
X_train = np.load("C:/Users/lggr1/OneDrive/Escritorio/Data Analysis/Module 2/data/p2_evaluation/X_train.npy")
X_test = np.load("C:/Users/lggr1/OneDrive/Escritorio/Data Analysis/Module 2/data/p2_evaluation/X_test.npy")
y_train = np.load("C:/Users/lggr1/OneDrive/Escritorio/Data Analysis/Module 2/data/p2_evaluation/y_train.npy")
y_test = np.load("C:/Users/lggr1/OneDrive/Escritorio/Data Analysis/Module 2/data/p2_evaluation/y_test.npy")

# Log-transform the data
X_train_log = np.log2(X_train + 1)
X_test_log = np.log2(X_test + 1)

# Logistic Regression Model
logreg = LogisticRegression(max_iter=10000, penalty='l2', solver='liblinear', C=10)
logreg.fit(X_train_log, y_train)

# Select top 100 features based on coefficients
model = SelectFromModel(logreg, max_features=100, prefit=True)
X_train_selected = model.transform(X_train_log)
X_test_selected = model.transform(X_test_log)

# Evaluate the model with selected features
logreg.fit(X_train_selected, y_train)
y_pred = logreg.predict(X_test_selected)
evaluation_accuracy = accuracy_score(y_test, y_pred)

# Evaluate with random features
random_indices = np.random.choice(X_train_log.shape[1], 100, replace=False)
X_train_random = X_train_log[:, random_indices]
X_test_random = X_test_log[:, random_indices]

logreg.fit(X_train_random, y_train)
y_pred_random = logreg.predict(X_test_random)
random_accuracy = accuracy_score(y_test, y_pred_random)

# Evaluate with high variance features
variances = np.var(X_train_log, axis=0)
high_variance_indices = np.argsort(variances)[-100:]
X_train_high_variance = X_train_log[:, high_variance_indices]
X_test_high_variance = X_test_log[:, high_variance_indices]

logreg.fit(X_train_high_variance, y_train)
y_pred_high_variance = logreg.predict(X_test_high_variance)
high_variance_accuracy = accuracy_score(y_test, y_pred_high_variance)

# Plot histograms of feature variances for random and high-variance features
plt.figure(figsize=(10, 6))
plt.hist(np.var(X_train_log[:, random_indices], axis=0), bins=30, alpha=0.5, label='Random Features')
plt.hist(np.var(X_train_log[:, high_variance_indices], axis=0), bins=30, alpha=0.5, label='High-Variance Features')
plt.xlabel('Feature Variance')
plt.ylabel('Frequency')
plt.title('Histogram of Feature Variances')
plt.legend()
plt.show()

# Print accuracies
print(f'Evaluation accuracy with selected features: {evaluation_accuracy}')
print(f'Evaluation accuracy with random features: {random_accuracy}')
print(f'Evaluation accuracy with high-variance features: {high_variance_accuracy}')
