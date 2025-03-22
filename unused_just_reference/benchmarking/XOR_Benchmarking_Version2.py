import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# XOR input and target data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)


import tensorflow as tf
import numpy as np

def custom_glorot_uniform(shape, dtype=tf.float32, seed=None):
    """
    Custom Glorot Uniform initializer as a callable function.

    Args:
        shape: Shape of the tensor to initialize.
        dtype: Data type of the tensor.
        seed: Optional random seed.

    Returns:
        A tensor initialized with values drawn from a uniform distribution.
    """
    if len(shape) < 2:
        fan_in = fan_out = 1
    else:
        fan_in = np.prod(shape[:-1])
        fan_out = shape[-1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random.uniform(shape, -limit, limit, dtype=dtype, seed=seed)



# Define model architecture matching yours: 2 inputs -> 2 hidden (tanh) -> 1 output (linear)
def create_model(initializer):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2, activation='tanh', input_shape=(2,),
                             kernel_initializer=initializer,
                             bias_initializer=initializer),
        tf.keras.layers.Dense(1, activation='linear',
                             kernel_initializer=initializer,
                             bias_initializer=initializer)
    ])
    return model





def train_until_convergence(model, X, y, max_epochs=1000, threshold=0.5):
    """
    Trains the model one epoch at a time until the predictions
    correctly match the target labels or until max_epochs is reached.

    Args:
        model: The Keras model to train.
        X: Input data.
        y: Target data.
        max_epochs: Maximum number of epochs to run.
        threshold: Threshold for binary decision.

    Returns:
        The number of epochs run.
    """
    epochs_run = 0
    for epoch in range(max_epochs):
        # Train for a single epoch
        model.fit(X, y, epochs=1, verbose=0)
        epochs_run += 1

        # Check predictions after this epoch
        predictions = model.predict(X, verbose=0)
        binary_preds = (predictions > threshold).astype(int)
        if np.array_equal(binary_preds, y):
            break
    return epochs_run




# Optimizers to test
optimizers = {
    'SGD': tf.keras.optimizers.SGD(learning_rate=0.1),
    'SGD_momentum': tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
    'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=0.01),
    'Adam': tf.keras.optimizers.Adam(learning_rate=0.01),
    'Adagrad': tf.keras.optimizers.Adagrad(learning_rate=0.1),
    'Adamax': tf.keras.optimizers.Adamax(learning_rate=0.01),
    'Nadam': tf.keras.optimizers.Nadam(learning_rate=0.01)
}

# Number of runs per optimizer
num_runs = 100
max_epochs = 1000  # Maximum epochs before stopping

# Dictionary to store results
results = defaultdict(list)
convergence_counts = defaultdict(int)

xavier_init = custom_glorot_uniform  # Use the function as the initializer


# Run benchmark
for optimizer_name, optimizer in optimizers.items():
    print(f"Testing {optimizer_name}...")

    for run in range(num_runs):
        # Create fresh model with Xavier init
        model = create_model(xavier_init)

        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Train model until convergence (or max_epochs reached)
        start_time = time.time()
        epochs_to_converge = train_until_convergence(model, X, y, max_epochs, threshold=0.5)
        end_time = time.time()

        # Store results
        results[optimizer_name].append(epochs_to_converge)

        if epochs_to_converge < max_epochs:
            convergence_counts[optimizer_name] += 1



# Calculate statistics
stats = {}
for optimizer_name, epochs_list in results.items():
    converged_epochs = [e for e in epochs_list if e < max_epochs]
    if converged_epochs:
        stats[optimizer_name] = {
            'mean': np.mean(converged_epochs),
            'median': np.median(converged_epochs),
            'min': np.min(converged_epochs),
            'max': np.max(converged_epochs),
            'std': np.std(converged_epochs),
            'convergence_rate': convergence_counts[optimizer_name] / num_runs * 100
        }
    else:
        stats[optimizer_name] = {
            'mean': float('inf'),
            'median': float('inf'),
            'min': float('inf'),
            'max': float('inf'),
            'std': 0,
            'convergence_rate': 0
        }

# Sort results by average epochs to converge
sorted_stats = sorted(stats.items(), key=lambda x: x[1]['mean'])

# Print results
print("\nOptimizer Performance Summary (100 runs each):")
print("="*80)
print(f"{'Optimizer':<15} {'Mean':<8} {'Median':<8} {'Min':<5} {'Max':<5} {'StdDev':<8} {'Conv %':<8}")
print("-"*80)
for optimizer_name, stat in sorted_stats:
    print(f"{optimizer_name:<15} {stat['mean']:<8.1f} {stat['median']:<8.1f} {stat['min']:<5.0f} {stat['max']:<5.0f} {stat['std']:<8.1f} {stat['convergence_rate']:<8.1f}%")

# Plot distribution of convergence epochs
plt.figure(figsize=(12, 8))
ax = plt.subplot(111)

# Create boxplot
box_data = []
labels = []
for optimizer_name, _ in sorted_stats:
    # Only use data from runs that converged
    converged_data = [e for e in results[optimizer_name] if e < max_epochs]
    if converged_data:
        box_data.append(converged_data)
        labels.append(f"{optimizer_name}\n(mean: {stats[optimizer_name]['mean']:.1f})")

if box_data:
    bp = ax.boxplot(box_data, patch_artist=True)

    # Set colors
    colors = ['lightblue', 'lightgreen', 'salmon', 'lightyellow', 'lightcoral', 'lightcyan', 'lightpink']
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i % len(colors)])

    # Set plot properties
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel('Epochs to Converge')
    plt.title('XOR Problem: Epochs to Convergence by Optimizer (100 runs)')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Annotate with min values
    for i, optimizer_name in enumerate(labels):
        opt_name = optimizer_name.split('\n')[0]
        min_val = stats[opt_name]['min']
        plt.annotate(f"Min: {min_val}",
                    xy=(i+1, min_val),
                    xytext=(i+1, min_val - 5),
                    ha='center')

plt.tight_layout()
plt.savefig('optimizer_comparison.png')
plt.show()

print("\nBenchmark complete. Results show the number of epochs required to correctly")
print("classify all 4 XOR patterns using different optimizers with Xavier initialization.")