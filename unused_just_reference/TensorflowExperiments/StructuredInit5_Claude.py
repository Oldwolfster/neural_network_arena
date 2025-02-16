import tensorflow as tf
import numpy as np
import os
import random
from typing import List, Tuple

def create_model(layer_sizes: List[int], learning_rate: float = 0.1) -> tf.keras.Model:
    """Create a model with specified layer sizes."""
    layers = []
    for i, size in enumerate(layer_sizes):
        if i == 0:
            layers.append(tf.keras.layers.Dense(size, activation="tanh", input_shape=(2,)))
        else:
            layers.append(tf.keras.layers.Dense(size, activation="tanh" if i < len(layer_sizes)-1 else "sigmoid"))

    model = tf.keras.Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                 loss="mse",
                 metrics=["accuracy"])
    return model

def zero_weights_after_first(model: tf.keras.Model) -> None:
    """Zero out weights for all layers after the first hidden layer."""
    first_layer = True
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            if first_layer:
                first_layer = False
                continue
            weights, biases = layer.get_weights()
            weights = np.zeros_like(weights)
            layer.set_weights([weights, biases])

def run_experiment(seed_value: int,
                  layer_sizes: List[int],
                  learning_rate: float = 0.1,
                  epochs: int = 100) -> Tuple[float, float, dict, dict]:
    """Run experiment with given parameters."""
    # Set seeds
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)
    tf.keras.utils.set_random_seed(seed_value)

    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    # Create and train models
    model_random = create_model(layer_sizes, learning_rate)
    model_struct = create_model(layer_sizes, learning_rate)
    zero_weights_after_first(model_struct)

    history_random = model_random.fit(X, y, epochs=epochs, verbose=0)
    history_struct = model_struct.fit(X, y, epochs=epochs, verbose=0)

    return (history_random.history["loss"][-1],
            history_struct.history["loss"][-1],
            history_random.history,
            history_struct.history)

# Test configurations
architectures = [
    [2, 1],           # Original architecture
    [4, 1],           # Wider first layer
    [2, 2, 1],        # Additional hidden layer
    [4, 3, 1],        # Wider with extra hidden layer
    [4, 3, 2, 1],     # Deep network
]

learning_rates = [0.01, 0.1, 0.5]
seeds_per_config = 1

results = []
for arch in architectures:
    for lr in learning_rates:
        arch_results = []
        for seed in range(seeds_per_config):
            print(f"Running Seed: {seed}\tLR:{lr}\tarchitecture{arch}")
            random_loss, struct_loss, random_hist, struct_hist = run_experiment(
                seed, arch, lr)
            improvement = (1 - struct_loss / random_loss) * 100
            arch_results.append({
                'seed': seed,
                'random_loss': random_loss,
                'struct_loss': struct_loss,
                'improvement': improvement,
                'architecture': arch,
                'learning_rate': lr
            })
        results.extend(arch_results)

        # Calculate statistics for this configuration
        improvements = [r['improvement'] for r in arch_results]
        mean_improvement = np.mean(improvements)
        std_improvement = np.std(improvements)
        success_rate = np.mean([i > 0 for i in improvements]) * 100

        print(f"\nArchitecture: {arch}, Learning Rate: {lr}")
        print(f"Mean improvement: {mean_improvement:.2f}% (±{std_improvement:.2f})")
        print(f"Success rate: {success_rate:.1f}%")