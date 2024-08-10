import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate synthetic dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)

for i in range(len(X)):
    print(f"Sample {i+1}: X = {X[i]}, y = {y[i]}")

# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', marker='o')
plt.xlabel('Feature 1 (Weight)')
plt.ylabel('Feature 2 (Height)')
plt.title('Synthetic Dataset: Weight vs. Height')
plt.show()
