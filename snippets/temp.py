import matplotlib.pyplot as plt
import numpy as np

# Create a number line from 0 to 1
plt.figure(figsize=(8, 2))
plt.axhline(0, color='black', linewidth=1)
plt.xlim(0, 1)
plt.xticks(np.arange(0, 1.1, 0.1))

# Mark the decision boundary at 0.5
plt.scatter(0.5, 0, color='red', zorder=5)
plt.text(0.5, 0.02, 'Decision Boundary', horizontalalignment='center', color='red')

# Add labels
plt.text(0, -0.1, '0', horizontalalignment='center')
plt.text(1, -0.1, '1', horizontalalignment='center')

# Remove y-axis and spines
plt.gca().get_yaxis().set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Show the plot
plt.show()
