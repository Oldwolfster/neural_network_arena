# ChatGPT, this x axis ranges from 0 to 6 with 3 in the middle.  to match the battle, it would be better if .01 was in the middle and 0 was all the way

import numpy as np
import matplotlib.pyplot as plt

def loss_function(w):
    """Simulate a loss function J(w,b) with respect to w."""
    return 2 * (w - 3)**2 + 1

# Generate weight values
weights = np.linspace(0, 6, 100)

# Calculate corresponding loss values
losses = [loss_function(w) for w in weights]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(weights, losses, 'b-', linewidth=2)
plt.title('Overshooting Due to Too Large a Learning Rate', fontsize=16)
plt.xlabel('Weight', fontsize=14)
plt.ylabel('Loss (How Far Off Is Weight)', fontsize=14)

# Add points and arrows to simulate gradient descent
high_lr_points = [0.5, 5.5, 1, 5, 1.5, 4.5, 2, 4, 2.5, 3.5]
low_lr_points = [2.7, 2.8, 2.9, 3.0, 3.1]

# Plot high learning rate
for i in range(0, len(high_lr_points), 2):
    plt.annotate('', xy=(high_lr_points[i+1], loss_function(high_lr_points[i+1])),
                 xytext=(high_lr_points[i], loss_function(high_lr_points[i])),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=10))
    plt.plot(high_lr_points[i], loss_function(high_lr_points[i]), 'ro', markersize=8)

# Plot low learning rate
for i in range(len(low_lr_points) - 1):
    plt.annotate('', xy=(low_lr_points[i+1], loss_function(low_lr_points[i+1])),
                 xytext=(low_lr_points[i], loss_function(low_lr_points[i])),
                 arrowprops=dict(facecolor='purple', shrink=0.05, width=2, headwidth=10))
    plt.plot(low_lr_points[i], loss_function(low_lr_points[i]), 'mo', markersize=8)

# Add text annotations
plt.text(1, 12, r'$\alpha$ is too big', fontsize=12, color='red')
plt.text(4, 12, 'Use smaller ' + r'$\alpha$', fontsize=12, color='purple')

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()