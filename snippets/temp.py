import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
mean_time = 30  # mean time in minutes
std_dev = 5     # standard deviation in minutes
area = 100      # area in square meters

# Generate data points
x = np.linspace(mean_time - 4*std_dev, mean_time + 4*std_dev, 100)
y = stats.norm.pdf(x, mean_time, std_dev)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='Normal Distribution')
plt.fill_between(x, y, color='lightblue', alpha=0.7)

# Add labels and title
plt.title(f'Normal Distribution of Time to Sweep {area} sq meters', fontsize=16)
plt.xlabel('Time (minutes)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)

# Add vertical lines for mean and standard deviations
plt.axvline(mean_time, color='r', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(mean_time + std_dev, color='g', linestyle=':', linewidth=2, label='1 Std Dev')
plt.axvline(mean_time - std_dev, color='g', linestyle=':', linewidth=2)

# Add legend
plt.legend(fontsize=10)

# Add text annotations
plt.text(mean_time, max(y)*1.1, f'Mean: {mean_time} min', horizontalalignment='center', fontsize=10)
plt.text(mean_time + std_dev, max(y)*0.4, f'+1σ: {mean_time + std_dev} min', horizontalalignment='left', fontsize=10)
plt.text(mean_time - std_dev, max(y)*0.4, f'-1σ: {mean_time - std_dev} min', horizontalalignment='right', fontsize=10)

# Show the plot
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()