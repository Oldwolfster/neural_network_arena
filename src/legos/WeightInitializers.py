import numpy as np

class WeightInitializer:
    """Encapsulates weight initialization strategies with proper bias handling."""

    def __init__(self, method, bias_method=None, name="Custom", desc="", when_to_use="", best_for=""):
        self.method = method  # Function for weight initialization
        self.bias_method = bias_method if bias_method else lambda: np.random.uniform(-1, 1)  # Default uniform bias
        self.name = name
        self.desc = desc
        self.when_to_use = when_to_use
        self.best_for = best_for  # Best activation functions

    def __call__(self, shape):
        """Generates initialized weights & bias given a shape."""
        weights = self.method(shape)  # Generate weights
        bias = self.bias_method()  # Generate bias using the selected method
        return weights, bias
    def __repr__(self):
        """Custom representation for debugging."""
        return f"WeightInitializer(name={self.name})"

# 🔹 **1. Uniform Random Initialization (Default)**
Initializer_Uniform = WeightInitializer(
    method=lambda shape: np.random.uniform(-1, 1, shape),
    bias_method=lambda: np.random.uniform(-1, 1),  # Bias follows uniform distribution
    name="Uniform Random",
    desc="Assigns weights randomly from a uniform distribution [-1,1].",
    when_to_use="Useful for quick experimentation but may not be optimal for deep networks.",
    best_for="General use, but suboptimal for deep layers."
)

# 🔹 **2. Normal Distribution Initialization**
Initializer_Normal = WeightInitializer(
    method=lambda shape: np.random.normal(0, 1, shape),
    bias_method=lambda: np.random.normal(0, 1),  # Bias follows normal distribution
    name="Normal Random",
    desc="Assigns weights using a normal distribution (mean=0, std=1).",
    when_to_use="Works well in simple networks but can lead to exploding/vanishing gradients in deep models.",
    best_for="General use."
)

# 🔹 **3. Xavier (Glorot) Initialization** - Optimized for **sigmoid/tanh**
Initializer_Xavier = WeightInitializer(
    method=lambda shape: np.random.uniform(-np.sqrt(6 / sum(shape)), np.sqrt(6 / sum(shape)), shape),
    bias_method=lambda: np.random.uniform(-np.sqrt(6 / 2), np.sqrt(6 / 2)),  # Bias follows same scaling as weights
    name="Xavier (Glorot)",
    desc="Scales weights based on number of inputs/outputs to maintain signal propagation.",
    when_to_use="Good for Sigmoid/Tanh activations in shallow networks.",
    best_for="Sigmoid, Tanh."
) #    pitfalls="Breaks with ReLU-based activations. Use He instead.",

# 🔹 **4. He (Kaiming) Initialization** - Optimized for **ReLU-based activations**
Initializer_He = WeightInitializer(
    method=lambda shape: np.random.normal(0, np.sqrt(2 / shape[0]), shape),
    bias_method=lambda: np.random.normal(0, np.sqrt(2 / 2)),  # Bias follows weight scaling
    name="He (Kaiming)",
    desc="Optimized for ReLU, helps mitigate dying neurons.",
    when_to_use="Best for deep networks using ReLU to prevent vanishing gradients.",
    best_for="ReLU, Leaky ReLU."
)

# 🔹 **5. Small Random Values (Near Zero)**
Initializer_Tiny = WeightInitializer(
    method=lambda shape: np.random.randn(*shape) * 0.01,
    bias_method=lambda: np.random.randn() * 0.01,  # Bias also small
    name="Small Random",
    desc="Weights are initialized close to zero.",
    when_to_use="Used in some gradient-free methods or fine-tuning models.",
    best_for="Any activation."
)

# 🔹 **6. LeCun Initialization** - Optimized for **SELU activation**
Initializer_LeCun = WeightInitializer(
    method=lambda shape: np.random.normal(0, np.sqrt(1 / shape[0]), shape),
    bias_method=lambda: np.random.normal(0, np.sqrt(1 / 2)),  # Bias scaled similarly
    name="LeCun",
    desc="Similar to Xavier but optimized for self-normalizing networks.",
    when_to_use="Best when using SELU activation.",
    best_for="SELU, Tanh."
)
# 🔹 **7. Like my johnson **
Initializer_Huge = WeightInitializer(
    method=lambda shape: np.random.randn(*shape) * 10000.01+1111,
    bias_method=lambda: np.random.randn() * 0.01,  # Bias also small
    name="Large Random",
    desc="Weights are probably big.",
    when_to_use="Testing.",
    best_for="Any activation."
)

def xavier_kill_relu(shape):
    return np.random.uniform(-2, -1, size=shape)

Initializer_KillRelu = WeightInitializer(
    method=xavier_kill_relu,
    bias_method=lambda: -1.0,
    name="Xavier Kill ReLU",
    desc="Creates high chance of dead ReLU via negative initialization",
    when_to_use="Testing dead ReLU scenarios",
    best_for="Diagnostics"
)

