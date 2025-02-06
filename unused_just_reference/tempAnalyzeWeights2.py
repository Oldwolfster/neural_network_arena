import numpy as np

# Scaled features (without bias term)
X = np.array([
    [0, 1],
    [0.5, 0.475],
    [1, 0]
])

# Targets
y = np.array([200, 150, 110])

# Normal equation solution
# w = (X^T * X)^-1 * X^T * y
w = np.linalg.inv(X.T @ X) @ X.T @ y

print("Optimal weights:", w)

# Let's verify the solution
predictions = X @ w
print("\nPredictions:", predictions)
print("Actual targets:", y)
print("\nMean Squared Error:", np.mean((predictions - y)**2))