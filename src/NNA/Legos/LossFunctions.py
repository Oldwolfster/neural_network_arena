from src.NNA.Legos.ActivationFunctions import *
from src.NNA.engine.Neuron   import Neuron
import math

def _get_n(y_true):
    """
    Helper function to determine the number of samples in y_true.
    Returns 1 if y_true is a scalar (non-iterable), otherwise returns the length.
    """
    try:
        return len(y_true)
    except TypeError:
        return 1

class StrategyLossFunction:
    """
    🚀 Encapsulates loss function strategies with optional gradient computation.

    Attributes:
        loss: A function that computes the loss given predictions and true values.
        derivative: A function that computes the gradient of the loss, if available.
        name: The name of the loss function.
        short_name: Short version of name
        desc: A description of the loss function.
        when_to_use: Guidance on when to use this loss function.
        best_for: The scenarios or tasks where this loss function performs best.
        derivative_formula: A string representation of the derivative formula.
        allowed_activations: First value is used as default.  if gladiator tries to set to one not in list error is thrown.
        bd_rules: tuple containing up to 4 elements to define Binary Decision (BD) behavior.
            1) target_alpha (float): Default numerical target for Class Alpha (e.g., 0.0). Used for error calculation.
            2) target_beta (float): Default numerical target for Class Beta (e.g., 1.0). Used for error calculation.
            3) locked_targets_msg (str, optional):
                - If empty, target values can be freely changed by the model.
                - If it starts with "Error:", raise an error if the model tries to override targets.
                - Otherwise, raise a warning if modified.
            4) locked_threshold_msg (str, optional):
                - If empty, threshold can be freely changed.
                - If it starts with "Error:", raise an error if the model tries to override it.
                - Otherwise, raise a warning if modified.
            Threshold is assumed to bisect the two targets unless explicitly stated otherwise.
    """

    def __init__(
        self,
        loss,
        derivative=None,
        name="Custom",
        short_name="Custom",
        desc="",
        when_to_use="",
        best_for="",
        derivative_formula="",
        allowed_activations=None,   # 🚀 New: List of valid activation functions
    ):
        self.loss                           = loss
        self.derivative                     = derivative
        self.name                           = name
        self.short_name                     = short_name
        self.desc                           = desc
        self.when_to_use                    = when_to_use
        self.best_for                       = best_for
        self.derivative_formula             = derivative_formula
        self.allowed_activation_functions   = allowed_activations
        self.recommended_hidden_activations = [Activation_ReLU]

    def __call__(self, y_pred, y_true):
        """
        Computes the loss given predictions and true target values.
        """
        return self.loss(y_pred, y_true)

    def __repr__(self):
        """Custom representation for debugging."""
        return self.name

    def grad(self, y_pred, y_true):
        """
        Computes the gradient of the loss with respect to predictions.
        Raises NotImplementedError if no derivative provided.
        """
        if self.derivative is None:
            raise NotImplementedError("Gradient function not implemented for this loss function")
        return self.derivative(y_pred, y_true)

    @property
    def recommended_output_activation(self):
        if self.allowed_activation_functions:
            return self.allowed_activation_functions[0]
        return Activation_NoDamnFunction  # Safe fallback

    def validate_activation_functions(self):
        """
        Ensures the Gladiator is using a valid activation function setup for this loss function.
        """
        return  # right now it's more like spam


# 🔹 **1. Mean Squared Error (MSE) Loss**
def mse_loss(y_pred, y_true):
    """
    Computes the Mean Squared Error (MSE) loss.
    """
    n = _get_n(y_true)
    diffs = [(p - t)**2 for p, t in zip(y_pred, y_true)] if n > 1 else [(y_pred - y_true)**2]
    return sum(diffs) / n

def mse_derivative(y_pred, y_true):
    """
    Computes the derivative of the MSE loss with respect to predictions.
    """
    # Originally: n = y_true.shape[0] if isinstance(y_true, np.ndarray) else len(y_true)
    n = _get_n(y_true)
    if n > 1:
        return [2 * (p - t) / n for p, t in zip(y_pred, y_true)]
    return 2 * (y_pred - y_true) / n

Loss_MSE = StrategyLossFunction(
    loss               = mse_loss,
    derivative         = mse_derivative,
    name               = "Mean Squared Error",
    short_name         = "MSE",
    desc               = "Calculates the average of the squares of differences between predictions and actual values.",
    when_to_use        = "Commonly used for regression problems.",
    best_for           = "Regression tasks.",
    allowed_activations= None,
    derivative_formula = "2 * (prediction - target)"
)

# 🔹 **2. Mean Absolute Error (MAE) Loss**
def mae_loss(y_pred, y_true):
    """
    Computes the Mean Absolute Error (MAE) loss.
    """
    n = _get_n(y_true)
    diffs = [abs(p - t) for p, t in zip(y_pred, y_true)] if n > 1 else [abs(y_pred - y_true)]
    return sum(diffs) / n

def mae_derivative(y_pred, y_true):
    """
    Computes the derivative of the MAE loss with respect to predictions.
    Note: The derivative is undefined at zero; sign logic is used here.
    """
    # Originally: n = y_true.shape[0] if isinstance(y_true, np.ndarray) else len(y_true)
    n = _get_n(y_true)
    if n > 1:
        return [ (1 if p > t else -1 if p < t else 0) / n for p, t in zip(y_pred, y_true) ]
    return (1 if y_pred > y_true else -1 if y_pred < y_true else 0) / n

Loss_MAE = StrategyLossFunction(
    loss               = mae_loss,
    derivative         = mae_derivative,
    name               = "Mean Absolute Error",
    short_name         = "MAE",
    desc               = "Calculates the average of the absolute differences between predictions and actual values.",
    when_to_use        = "Useful for regression tasks less sensitive to outliers.",
    best_for           = "Regression tasks with outlier presence.",
    allowed_activations= None,
    derivative_formula = "sign(prediction - target) / n"
)

# 🔹 **7. Binary Cross-Entropy with Logits (BCEWithLogits) Loss**
def bce_with_logits_loss(logits, y_true):
    """
    Numerically stable BCE loss using raw logits instead of Sigmoid outputs.
    """
    n = _get_n(y_true)
    vals = []
    preds = logits if n > 1 else [logits]
    trues = y_true if n > 1 else [y_true]
    for z, t in zip(preds, trues):
        vals.append(max(z, 0) - z * t + math.log(1 + math.exp(-abs(z))))
    return sum(vals) / n

def bce_with_logits_derivative(logits, y_true):
    """
    Computes ∂L/∂logits = sigmoid(logits) - y_true
    """
    n = _get_n(y_true)
    sigmoid = lambda z: 1 / (1 + math.exp(-z))
    if n > 1:
        return [sigmoid(z) - t for z, t in zip(logits, y_true)]
    return sigmoid(logits) - y_true

Loss_BCEWithLogits = StrategyLossFunction(
    loss               = bce_with_logits_loss,
    derivative         = bce_with_logits_derivative,
    name               = "Binary Cross-Entropy with Logits",
    short_name         = "BCE_WL",
    desc               = "Numerically stable BCE loss using raw logits instead of Sigmoid outputs.",
    when_to_use        = "Use when working with raw logits (no Sigmoid in final layer).",
    best_for           = "Binary classification tasks with logits.",
    derivative_formula = "sigmoid(logits) - target",
    allowed_activations= None
)

# 🔹 **3. Binary Cross-Entropy (BCE) Loss**
def _clip(x, low, high):
    return high if x > high else low if x < low else x

def binary_crossentropy_loss(y_pred, y_true, epsilon=1e-15):
    """
    Computes the Binary Cross-Entropy loss for binary classification.
    """
    n = _get_n(y_true)
    preds = y_pred if n > 1 else [y_pred]
    trues = y_true if n > 1 else [y_true]
    losses = []
    for p, t in zip(preds, trues):
        pc = _clip(p, epsilon, 1 - epsilon)
        losses.append(-(t * math.log(pc) + (1 - t) * math.log(1 - pc)))
    return sum(losses) / n

def binary_crossentropy_derivative(y_pred, y_true, epsilon=1e-15):
    """
    Computes the derivative of the Binary Cross-Entropy loss.
    """
    n = _get_n(y_true)
    preds = y_pred if n > 1 else [y_pred]
    trues = y_true if n > 1 else [y_true]
    grads = []
    for p, t in zip(preds, trues):
        pc = _clip(p, epsilon, 1 - epsilon)
        grads.append(-(t/pc - (1 - t)/(1 - pc)) / n)
    return grads if n > 1 else grads[0]

Loss_BCE = StrategyLossFunction(
    loss               = binary_crossentropy_loss,
    derivative         = binary_crossentropy_derivative,
    name               = "Binary Cross-Entropy",
    short_name         = "BCE",
    desc               = "Calculates loss for binary classification using cross-entropy.",
    when_to_use        = "Ideal for binary classification problems.",
    best_for           = "Binary classification.",
    derivative_formula = "- (target / prediction - (1 - target) / (1 - prediction)) / n",
    allowed_activations= [Activation_Sigmoid]
)

# 🔹 **5. Hinge Loss**
def hinge_loss(y_pred, y_true):
    """
    Computes the Hinge loss. Assumes y_true is +1 or -1.
    """
    n      = _get_n(y_true)
    preds  = y_pred if n > 1 else [y_pred]
    trues  = y_true if n > 1 else [y_true]
    vals   = [max(0, 1 - t * p) for p, t in zip(preds, trues)]
    return sum(vals) / n

def hinge_derivative(y_pred, y_true):
    """
    Derivative of the Hinge loss.
    """
    n      = _get_n(y_true)
    preds  = y_pred if n > 1 else [y_pred]
    trues  = y_true if n > 1 else [y_true]
    grads  = []
    for p, t in zip(preds, trues):
        margin = 1 - t * p
        grad   = -t * margin if margin > 0 else 0
        grads.append(grad / n)
    return grads if n > 1 else grads[0]

Loss_Hinge = StrategyLossFunction(
    loss               = hinge_loss,
    derivative         = hinge_derivative,
    name               = "Hinge Loss",
    short_name         = "Hinge",
    desc               = "Used primarily for maximum-margin classification (e.g., SVMs).",
    when_to_use        = "Useful for support vector machines and related models.",
    best_for           = "Binary classification with margin-based methods.",
    derivative_formula = "where(1 - target * prediction > 0, -target * margin, 0) / n",
    allowed_activations= [Activation_NoDamnFunction]
)

# 🔹 **6. Log-Cosh Loss**
def logcosh_loss(y_pred, y_true):
    """
    Computes the Log-Cosh loss.
    """
    n      = _get_n(y_true)
    preds  = y_pred if n > 1 else [y_pred]
    trues  = y_true if n > 1 else [y_true]
    vals   = [math.log(math.cosh(p - t)) for p, t in zip(preds, trues)]
    return sum(vals) / n

def logcosh_derivative(y_pred, y_true):
    """
    Computes the derivative of the Log-Cosh loss.
    """
    n      = _get_n(y_true)
    preds  = y_pred if n > 1 else [y_pred]
    trues  = y_true if n > 1 else [y_true]
    grads  = [math.tanh(p - t) / n for p, t in zip(preds, trues)]
    return grads if n > 1 else grads[0]

Loss_LogCosh = StrategyLossFunction(
    loss               = logcosh_loss,
    derivative         = logcosh_derivative,
    name               = "Log-Cosh Loss",
    short_name         = "LCL",
    desc               = "Calculates loss using the logarithm of the hyperbolic cosine of the prediction error.",
    when_to_use        = "A smooth loss function less sensitive to outliers than MSE.",
    best_for           = "Regression tasks.",
    allowed_activations= [Activation_NoDamnFunction, Activation_Tanh, Activation_ReLU, Activation_LeakyReLU],
    derivative_formula = "tanh(prediction - target) / n"
)

# 🔹 **Huber Loss**
def huber_loss(y_pred, y_true, delta=1.0):
    """
    Computes the Huber loss.
    - Quadratic for |error| ≤ delta, linear beyond.
    """
    n      = _get_n(y_true)
    preds  = y_pred if n > 1 else [y_pred]
    trues  = y_true if n > 1 else [y_true]
    vals   = []
    for p, t in zip(preds, trues):
        err = p - t
        if abs(err) <= delta:
            vals.append(0.5 * err**2)
        else:
            vals.append(delta * (abs(err) - 0.5 * delta))
    return sum(vals) / n

def huber_derivative(y_pred, y_true, delta=1.0):
    """
    Derivative of the Huber loss wrt predictions.
    """
    n      = _get_n(y_true)
    preds  = y_pred if n > 1 else [y_pred]
    trues  = y_true if n > 1 else [y_true]
    grads  = []
    for p, t in zip(preds, trues):
        err = p - t
        grad = err if abs(err) <= delta else delta * (1 if err > 0 else -1)
        grads.append(grad / n)
    return grads if n > 1 else grads[0]

Loss_Huber = StrategyLossFunction(
    loss               = huber_loss,
    derivative         = huber_derivative,
    name               = "Huber Loss",
    short_name         = "Huber",
    desc               = (
        "Combines MSE and MAE: squared for small diffs, linear for large diffs to reduce outlier impact."
    ),
    when_to_use        = "Regression problems needing robustness to outliers.",
    best_for           = "Regression tasks with potential outliers.",
    allowed_activations= None,
    derivative_formula = "error if |error| ≤ δ else δ·sign(error)"
)

# 🔹 **Half Wit Error**
def half_wit_loss(y_pred, y_true):
    """
    Computes the same value as MSE — placeholder to match structure.
    """
    n     = _get_n(y_true)
    diffs = [(p - t)**2 for p, t in zip(y_pred, y_true)] if n > 1 else [(y_pred - y_true)**2]
    return sum(diffs) / n

def half_wit_derivative(y_pred, y_true):
    """
    Computes the raw error: (prediction - target) — skipping the 2x and division.
    """
    if _get_n(y_true) > 1:
        return [p - t for p, t in zip(y_pred, y_true)]
    return y_pred - y_true

Loss_HalfWit = StrategyLossFunction(
    loss               = half_wit_loss,
    derivative         = half_wit_derivative,
    name               = "Half Wit Error",
    short_name         = "HalfWit",
    desc               = "Returns the raw error instead of a true gradient. Half the math, all the charm.",
    when_to_use        = "When you want an honest answer.",
    best_for           = "Clarity-preferred situations.",
    allowed_activations= None,
    derivative_formula = "(prediction - target)"
)

__all__ = [name for name, obj in globals().items()
           if isinstance(obj, StrategyLossFunction)]
