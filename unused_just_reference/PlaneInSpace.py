import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compare_two_planes(params1, params2, noise_level=5, sample_size=100, input_range=(-10, 10)):
    """
    Compare the results of two sets of plane parameters side by side.
    Args:
        params1 (dict): First set of parameters (weights and bias).
        params2 (dict): Second set of parameters (weights and bias).
        noise_level (float): Standard deviation of noise for normal distribution.
        sample_size (int): Number of points to generate.
        input_range (tuple): Range of x and y values for inputs.
    """
    # Unpack parameters
    w1_1, w2_1, b1 = params1["w1"], params1["w2"], params1["bias"]
    w1_2, w2_2, b2 = params2["w1"], params2["w2"], params2["bias"]

    # Generate inputs
    x = np.random.uniform(input_range[0], input_range[1], sample_size)
    y = np.random.uniform(input_range[0], input_range[1], sample_size)

    # Generate true z-values with noise for both parameter sets
    z1 = w1_1 * x + w2_1 * y + b1 + np.random.normal(0, noise_level, sample_size)
    z2 = w1_2 * x + w2_2 * y + b2 + np.random.normal(0, noise_level, sample_size)

    # Create grids for the planes
    xx, yy = np.meshgrid(np.linspace(input_range[0], input_range[1], 10),
                         np.linspace(input_range[0], input_range[1], 10))
    zz1 = w1_1 * xx + w2_1 * yy + b1
    zz2 = w1_2 * xx + w2_2 * yy + b2

    # Plot side-by-side
    fig = plt.figure(figsize=(15, 7))

    # First Parameter Set
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(xx, yy, zz1, alpha=0.5, color='blue', edgecolor='k', rstride=100, cstride=100)
    ax1.scatter(x, y, z1, color='red', label='Data Points (Set 1)', alpha=0.7)
    ax1.set_title(f"Set 1: w1={w1_1}, w2={w2_1}, bias={b1}")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend()

    # Second Parameter Set
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(xx, yy, zz2, alpha=0.5, color='green', edgecolor='k', rstride=100, cstride=100)
    ax2.scatter(x, y, z2, color='orange', label='Data Points (Set 2)', alpha=0.7)
    ax2.set_title(f"Set 2: w1={w1_2}, w2={w2_2}, bias={b2}")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.legend()

    plt.show()

# Example Usage
if __name__ == "__main__":
    # Define two sets of parameters
    params1 = {"w1": 3, "w2": 2, "bias": 5}
    params2 = {"w1": 16, "w2": 2, "bias": 5}

    compare_two_planes(params1, params2, noise_level=5)
