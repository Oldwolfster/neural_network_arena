# Sample data for two features
x1 = [100, 200, 300, 400, 500]  # Large values
x2 = [0.1, 0.2, 0.3, 0.4, 0.5]  # Small values

# Min-Max scaling
x1_minmax = (x1 - min(x1)) / (max(x1) - min(x1))
x2_minmax = (x2 - min(x2)) / (max(x2) - min(x2))
# Both now range from 0 to 1: [0, 0.25, 0.5, 0.75, 1]

# Z-score
x1_zscore = (x1 - mean(x1)) / std(x1)
x2_zscore = (x2 - mean(x2)) / std(x2)
# Both now have mean=0, std=1: [-1.41, -0.71, 0, 0.71, 1.41]