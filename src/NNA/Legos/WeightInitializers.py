import random
import math

class ScalerWeightInitializer:
    """Encapsulates weight initialization strategies with proper bias handling."""

    def __init__(self, method, bias_method=None, name="Custom", desc="", when_to_use="", best_for=""):
        self.method = method  # Function for weight initialization
        self.bias_method = bias_method if bias_method else lambda: random.uniform(-1, 1)  # Default uniform bias
        self.name = name
        self.desc = desc
        self.when_to_use = when_to_use
        self.best_for = best_for  # Best activation functions

    def __call__(self, shape):
        """Generates initialized weights & bias given a shape."""
        weights = self.method(shape)  # Generate weights
        bias = self.bias_method()     # Generate bias using the selected method
        return weights, bias

    def __repr__(self):
        """Custom representation for debugging."""
        return self.name

def _dims(shape):
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)

def _generate(shape, generator):
    dims = _dims(shape)
    if len(dims) == 1:
        return [generator() for _ in range(dims[0])]
    elif len(dims) == 2:
        rows, cols = dims
        return [[generator() for _ in range(cols)] for _ in range(rows)]
    else:
        raise ValueError("Only 1D and 2D shapes supported")

def rand_uniform(low, high, shape):
    return _generate(shape, lambda: random.uniform(low, high))

def rand_normal(mean, std, shape):
    return _generate(shape, lambda: random.gauss(mean, std))

def rand_scalar_uniform(low=-1, high=1):
    return random.uniform(low, high)

def rand_scalar_normal(mean=0, std=1):
    return random.gauss(mean, std)

# 🔹 **1. Uniform Random Initialization (Default)**
Initializer_Uniform = ScalerWeightInitializer(
    method=lambda shape: rand_uniform(-1, 1, shape),
    bias_method=lambda: rand_scalar_uniform(-1, 1),  # Bias follows uniform distribution
    name="Uniform Random",
    desc="Assigns weights randomly from a uniform distribution [-1,1].",
    when_to_use="Useful for quick experimentation but may not be optimal for deep networks.",
    best_for="General use, but suboptimal for deep layers."
)

# 🔹 **2. Normal Distribution Initialization**
Initializer_Normal = ScalerWeightInitializer(
    method=lambda shape: rand_normal(0, 1, shape),
    bias_method=lambda: rand_scalar_normal(0, 1),  # Bias follows normal distribution
    name="Normal Random",
    desc="Assigns weights using a normal distribution (mean=0, std=1).",
    when_to_use="Works well in simple networks but can lead to exploding/vanishing gradients in deep models.",
    best_for="General use."
)

# 🔹 **3. Xavier (Glorot) Initialization** - Optimized for **sigmoid/tanh**
Initializer_Xavier = ScalerWeightInitializer(
    method=lambda shape: rand_uniform(
        -math.sqrt(6 / sum(_dims(shape))),
        math.sqrt(6 / sum(_dims(shape))),
        shape
    ),
    bias_method=lambda: rand_scalar_uniform(-math.sqrt(6 / 2), math.sqrt(6 / 2)),  # Bias follows same scaling as weights
    name="Xavier (Glorot)",
    desc="Scales weights based on number of inputs/outputs to maintain signal propagation.",
    when_to_use="Good for Sigmoid/Tanh activations in shallow networks.",
    best_for="Sigmoid, Tanh."
)

# 🔹 **4. He (Kaiming) Initialization** - Optimized for **ReLU-based activations**
Initializer_He = ScalerWeightInitializer(
    method=lambda shape: rand_normal(0, math.sqrt(2 / _dims(shape)[0]), shape),
    bias_method=lambda: rand_scalar_normal(0, math.sqrt(2 / 2)),  # Bias follows weight scaling
    name="He (Kaiming)",
    desc="Optimized for ReLU, helps mitigate dying neurons.",
    when_to_use="Best for deep networks using ReLU to prevent vanishing gradients.",
    best_for="ReLU, Leaky ReLU."
)

# 🔹 **5. Small Random Values (Near Zero)**
Initializer_Tiny = ScalerWeightInitializer(
    method=lambda shape: rand_normal(0, 0.01, shape),
    bias_method=lambda: rand_scalar_normal(0, 0.01),  # Bias also small
    name="Small Random",
    desc="Weights are initialized close to zero.",
    when_to_use="Used in some gradient-free methods or fine-tuning models.",
    best_for="Any activation."
)

# 🔹 **6. LeCun Initialization** - Optimized for **SELU activation**
Initializer_LeCun = ScalerWeightInitializer(
    method=lambda shape: rand_normal(0, math.sqrt(1 / _dims(shape)[0]), shape),
    bias_method=lambda: rand_scalar_normal(0, math.sqrt(1 / 2)),  # Bias scaled similarly
    name="LeCun",
    desc="Similar to Xavier but optimized for self-normalizing networks.",
    when_to_use="Best when using SELU activation.",
    best_for="SELU, Tanh."
)

# 🔹 **7. Like my johnson **
Initializer_Huge = ScalerWeightInitializer(
    method=lambda shape: _generate(shape, lambda: random.gauss(0, 1) * 10000.01 + 1111),
    bias_method=lambda: random.gauss(0, 0.01),  # Bias also small
    name="Large Random",
    desc="Weights are probably big.",
    when_to_use="Testing.",
    best_for="Any activation."
)

def xavier_kill_relu(shape):
    return rand_uniform(-2, -1, shape)

Initializer_KillRelu = ScalerWeightInitializer(
    method=xavier_kill_relu,
    bias_method=lambda: -1.0,
    name="Xavier Kill ReLU",
    desc="Creates high chance of dead ReLU via negative initialization",
    when_to_use="Testing dead ReLU scenarios",
    best_for="Diagnostics"
)
