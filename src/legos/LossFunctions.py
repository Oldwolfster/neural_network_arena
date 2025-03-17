import numpy as np

def _get_n(y_true):
    """
    Helper function to determine the number of samples in y_true.
    Returns 1 if y_true is a scalar (or 0-d array), otherwise returns the size of the first dimension.
    """
    if np.isscalar(y_true):
        return 1
    try:
        y_true_arr = np.array(y_true)
        if y_true_arr.ndim == 0:
            return 1
        return y_true_arr.shape[0]
    except TypeError:
        # If y_true is not iterable, treat it as scalar.
        return 1

class LossFunction:
    """
    🚀 Encapsulates loss function strategies with optional gradient computation.

    Attributes:
        loss: A function that computes the loss given predictions and true values.
        derivative: A function that computes the gradient of the loss, if available.
        name: The name of the loss function.
        desc: A description of the loss function.
        when_to_use: Guidance on when to use this loss function.
        best_for: The scenarios or tasks where this loss function performs best.
        derivative_formula: A string representation of the derivative formula.
    """
    def __init__(self, loss, derivative=None, name="Custom", desc="", when_to_use="", best_for="", derivative_formula=""):
        self.loss = loss  # Function to compute the loss.
        self.derivative = derivative  # Optional function to compute the gradient of the loss.
        self.name = name
        self.desc = desc
        self.when_to_use = when_to_use
        self.best_for = best_for
        self.derivative_formula = derivative_formula  # String representation of the derivative formula.

    def __call__(self, y_pred, y_true):
        """
        Computes the loss given predictions and true target values.

        Parameters:
            y_pred: The predicted values.
            y_true: The actual target values.

        Returns:
            The computed loss.
        """
        return self.loss(y_pred, y_true)

    def __repr__(self):
        """Custom representation for debugging."""
        return f"WeightInitializer(name={self.name})"

    def grad(self, y_pred, y_true):
        """
        Computes the gradient of the loss with respect to predictions.

        Parameters:
            y_pred: The predicted values.
            y_true: The actual target values.

        Returns:
            The computed gradient.

        Raises:
            NotImplementedError: If the derivative function is not provided.
        """
        if self.derivative is None:
            raise NotImplementedError("Gradient function not implemented for this loss function")
        return self.derivative(y_pred, y_true)


# 🔹 **1. Mean Squared Error (MSE) Loss**
def mse_loss(y_pred, y_true):
    """
    Computes the Mean Squared Error (MSE) loss.
    """
    return np.mean((y_pred - y_true) ** 2)

def mse_derivative(y_pred, y_true):
    """
    Computes the derivative of the MSE loss with respect to predictions.
    """
    # Originally: n = y_true.shape[0] if isinstance(y_true, np.ndarray) else len(y_true)
    n = _get_n(y_true)
    return 2 * (y_pred - y_true) / n

Loss_MSE = LossFunction(
    loss=mse_loss,
    derivative=mse_derivative,
    name="Mean Squared Error (MSE)",
    desc="Calculates the average of the squares of differences between predictions and actual values.",
    when_to_use="Commonly used for regression problems.",
    best_for="Regression tasks.",
    derivative_formula="2 * (prediction - target)"
)

# 🔹 **2. Mean Absolute Error (MAE) Loss**
def mae_loss(y_pred, y_true):
    """
    Computes the Mean Absolute Error (MAE) loss.
    """
    return np.mean(np.abs(y_pred - y_true))

def mae_derivative(y_pred, y_true):
    """
    Computes the derivative of the MAE loss with respect to predictions.
    Note: The derivative is undefined at zero; np.sign is used here.
    """
    # Originally: n = y_true.shape[0] if isinstance(y_true, np.ndarray) else len(y_true)
    n = _get_n(y_true)
    return np.sign(y_pred - y_true) / n

Loss_MAE = LossFunction(
    loss=mae_loss,
    derivative=mae_derivative,
    name="Mean Absolute Error (MAE)",
    desc="Calculates the average of the absolute differences between predictions and actual values.",
    when_to_use="Useful for regression tasks less sensitive to outliers.",
    best_for="Regression tasks with outlier presence.",
    derivative_formula="sign(prediction - target) / n"
)

# 🔹 **3. Binary Cross-Entropy (BCE) Loss**
def binary_crossentropy_loss(y_pred, y_true, epsilon=1e-15):
    """
    Computes the Binary Cross-Entropy loss for binary classification.
    Clipping is applied to avoid log(0) issues.
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_crossentropy_derivative(y_pred, y_true, epsilon=1e-15):
    """
    Computes the derivative of the Binary Cross-Entropy loss with respect to predictions.
    Clipping is applied to avoid division by zero.
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # Originally: n = y_true.shape[0] if isinstance(y_true, np.ndarray) else len(y_true)
    n = _get_n(y_true)
    return - (y_true / y_pred - (1 - y_true) / (1 - y_pred)) / n

Loss_BinaryCrossEntropy = LossFunction(
    loss=binary_crossentropy_loss,
    derivative=binary_crossentropy_derivative,
    name="Binary Cross-Entropy",
    desc="Calculates loss for binary classification tasks using cross-entropy.",
    when_to_use="Ideal for binary classification problems.",
    best_for="Binary classification.",
    derivative_formula="- (target / prediction - (1 - target) / (1 - prediction)) / n"
)

# 🔹 **4. Categorical Cross-Entropy Loss**
def categorical_crossentropy_loss(y_pred, y_true, epsilon=1e-15):
    """
    Computes the Categorical Cross-Entropy loss for multi-class classification.
    Assumes that y_true is one-hot encoded.
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # Sum over classes and average over samples.
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def categorical_crossentropy_derivative(y_pred, y_true):
    """
    Computes the derivative of the Categorical Cross-Entropy loss with respect to predictions.
    Assumes that y_pred is the output of a softmax layer.
    """
    # Originally: n = y_true.shape[0]
    n = _get_n(y_true)
    return (y_pred - y_true) / n

Loss_CategoricalCrossEntropy = LossFunction(
    loss=categorical_crossentropy_loss,
    derivative=categorical_crossentropy_derivative,
    name="Categorical Cross-Entropy",
    desc="Calculates loss for multi-class classification tasks using cross-entropy.",
    when_to_use="Ideal for multi-class classification problems with one-hot encoded targets.",
    best_for="Multi-class classification.",
    derivative_formula="(prediction - target) / n"
)

# 🔹 **5. Hinge Loss**
def hinge_loss(y_pred, y_true):
    """
    Computes the Hinge loss.
    Assumes y_true is encoded as +1 or -1.
    """
    return np.mean(np.maximum(0, 1 - y_true * y_pred))

def hinge_derivative(y_pred, y_true):
    """
    Computes the derivative of the Hinge loss with respect to predictions.
    """
    grad = np.where(1 - y_true * y_pred > 0, -y_true, 0)
    # Originally: n = y_true.shape[0] if isinstance(y_true, np.ndarray) else len(y_true)
    n = _get_n(y_true)
    return grad / n

Loss_Hinge = LossFunction(
    loss=hinge_loss,
    derivative=hinge_derivative,
    name="Hinge Loss",
    desc="Used primarily for maximum-margin classification (e.g., SVMs).",
    when_to_use="Useful for support vector machines and related models.",
    best_for="Binary classification with margin-based methods.",
    derivative_formula="where(1 - target * prediction > 0, -target, 0) / n"
)

# 🔹 **6. Log-Cosh Loss**
def logcosh_loss(y_pred, y_true):
    """
    Computes the Log-Cosh loss.
    """
    return np.mean(np.log(np.cosh(y_pred - y_true)))

def logcosh_derivative(y_pred, y_true):
    """
    Computes the derivative of the Log-Cosh loss with respect to predictions.
    """
    # Originally: n = y_true.shape[0] if isinstance(y_true, np.ndarray) else len(y_true)
    n = _get_n(y_true)
    return np.tanh(y_pred - y_true) / n

Loss_LogCosh = LossFunction(
    loss=logcosh_loss,
    derivative=logcosh_derivative,
    name="Log-Cosh Loss",
    desc="Calculates loss using the logarithm of the hyperbolic cosine of the prediction error.",
    when_to_use="A smooth loss function that is less sensitive to outliers than MSE.",
    best_for="Regression tasks.",
    derivative_formula="tanh(prediction - target) / n"
)

# 🔹 **7. Binary Cross-Entropy with Logits (BCEWithLogits) Loss**
def bce_with_logits_loss(logits, y_true):
    """
    Computes the Binary Cross-Entropy loss for binary classification using logits.
    This is numerically stable and does NOT require a Sigmoid activation beforehand.
    """
    return np.mean(np.maximum(logits, 0) - logits * y_true + np.log(1 + np.exp(-np.abs(logits))))

def bce_with_logits_derivative(logits, y_true):
    """
    Computes the gradient of Binary Cross-Entropy with Logits loss.
    Instead of using sigmoid explicitly, we use:
      ∂L/∂logits = σ(logits) - y_true
    """
    return (1 / (1 + np.exp(-logits))) - y_true  # Sigmoid(logits) - y_true

Loss_BCEWithLogits = LossFunction(
    loss=bce_with_logits_loss,
    derivative=bce_with_logits_derivative,
    name="Binary Cross-Entropy with Logits",
    desc="Numerically stable BCE loss using raw logits instead of Sigmoid outputs.",
    when_to_use="Use this instead of BCE when working with raw logits (no Sigmoid activation in the last layer).",
    best_for="Binary classification tasks where Sigmoid is removed from the model's final layer.",
    derivative_formula="sigmoid(logits) - target"
)
