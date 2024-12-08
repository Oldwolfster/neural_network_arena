import numpy as np
import matplotlib.pyplot as plt

# Unscaled data
X_unscaled = np.array([
    [0, 200],
    [50, 95],
    [100, 10]
])
y = np.array([200, 150, 110])

# Min-Max scaled data
def min_max_scale(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

X_scaled = min_max_scale(X_unscaled)

def compute_gradient(X, y, weights):
    n = len(y)
    predictions = X @ weights
    return -2/n * X.T @ (y - predictions)

# Gradient computation for unscaled and scaled data
def plot_gradient_comparison():
    plt.figure(figsize=(12,5))

    # Unscaled data gradient computation
    weights_unscaled = np.zeros(2)
    gradients_unscaled = []
    for _ in range(100):
        grad = compute_gradient(X_unscaled, y, weights_unscaled)
        gradients_unscaled.append(np.linalg.norm(grad))
        weights_unscaled -= 0.01 * grad

    # Scaled data gradient computation
    weights_scaled = np.zeros(2)
    gradients_scaled = []
    for _ in range(100):
        grad = compute_gradient(X_scaled, y, weights_scaled)
        gradients_scaled.append(np.linalg.norm(grad))
        weights_scaled -= 0.01 * grad

    # Plot gradients
    plt.subplot(1,2,1)
    plt.title('Unscaled Data Gradient Magnitude')
    plt.plot(gradients_unscaled)
    plt.ylabel('Gradient Magnitude')
    plt.xlabel('Iteration')

    plt.subplot(1,2,2)
    plt.title('Scaled Data Gradient Magnitude')
    plt.plot(gradients_scaled)
    plt.ylabel('Gradient Magnitude')
    plt.xlabel('Iteration')

    plt.tight_layout()
    plt.show()

    print("Unscaled final weights:", weights_unscaled)
    print("Scaled final weights:", weights_scaled)

# Run the comparison
plot_gradient_comparison()

# Detailed gradient analysis
def detailed_gradient_analysis():
    print("\nUnscaled Data Gradient:")
    grad_unscaled = compute_gradient(X_unscaled, y, np.zeros(2))
    print("Gradient for each feature:", grad_unscaled)

    print("\nScaled Data Gradient:")
    grad_scaled = compute_gradient(X_scaled, y, np.zeros(2))
    print("Gradient for each feature:", grad_scaled)

detailed_gradient_analysis()