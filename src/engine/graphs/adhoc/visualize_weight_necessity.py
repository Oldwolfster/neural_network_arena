import matplotlib.pyplot as plt
import numpy as np

def visualize_weight_necessity(figsize=(15, 5)):
    """
    Demonstrate why a linear model might need opposing weights
    even with positive inputs
    """
    # Create sample data where one feature should have negative weight
    np.random.seed(42)
    n_samples = 100

    # Create two input features
    x1 = np.random.uniform(0, 10, n_samples)  # First feature
    x2 = np.random.uniform(0, 10, n_samples)  # Second feature

    # Create target where x1 contributes positively but x2 should subtract
    # Example: target = 2*x1 - 0.5*x2 + noise
    y = 2*x1 - 0.5*x2 + np.random.normal(0, 1, n_samples)

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    # Plot 1: x1 vs y
    ax1.scatter(x1, y, alpha=0.5)
    ax1.set_title('Feature 1 vs Target\n(Positive Correlation)')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Target')

    # Add trend line
    z1 = np.polyfit(x1, y, 1)
    p1 = np.poly1d(z1)
    x1_trend = np.linspace(min(x1), max(x1), 100)
    ax1.plot(x1_trend, p1(x1_trend), "r--", alpha=0.8)
    ax1.text(0.05, 0.95, f'Correlation: {np.corrcoef(x1, y)[0,1]:.3f}',
             transform=ax1.transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    # Plot 2: x2 vs y
    ax2.scatter(x2, y, alpha=0.5)
    ax2.set_title('Feature 2 vs Target\n(Negative Correlation)')
    ax2.set_xlabel('Feature 2')

    # Add trend line
    z2 = np.polyfit(x2, y, 1)
    p2 = np.poly1d(z2)
    x2_trend = np.linspace(min(x2), max(x2), 100)
    ax2.plot(x2_trend, p2(x2_trend), "r--", alpha=0.8)
    ax2.text(0.05, 0.95, f'Correlation: {np.corrcoef(x2, y)[0,1]:.3f}',
             transform=ax2.transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    # Plot 3: Actual vs Predicted with both features
    X = np.column_stack((x1, x2))
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    ax3.scatter(y, y_pred, alpha=0.5)
    ax3.plot([min(y), max(y)], [min(y), max(y)], 'r--', alpha=0.8)
    ax3.set_title('Actual vs Predicted\nUsing Both Features')
    ax3.set_xlabel('Actual')
    ax3.set_ylabel('Predicted')
    ax3.text(0.05, 0.95, f'Weight 1: {model.coef_[0]:.3f}\nWeight 2: {model.coef_[1]:.3f}',
             transform=ax3.transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig, (ax1, ax2, ax3)

# Example usage:

fig, axes = visualize_weight_necessity()
plt.show()
