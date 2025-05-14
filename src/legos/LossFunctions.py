import numpy as np

from src.Legos.ActivationFunctions import *
from src.engine.Neuron import Neuron


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
        bd_defaults = None          #Class A, Class B, Threshold
    ):
        self.loss               = loss  # Function to compute the loss.
        self.derivative         = derivative  # Optional function to compute the gradient of the loss.
        self.name               = name
        self.short_name         = short_name
        self.desc               = desc
        self.when_to_use        = when_to_use
        self.best_for           = best_for
        self.derivative_formula = derivative_formula  # String representation of the derivative formula.

        self.bd_defaults        = bd_defaults
        self.allowed_activation_functions       = allowed_activations #if allowed_activations is not None else [] # Store allowed activation functions (default = allow all)
        self.recommended_hidden_activations     = [Activation_ReLU]


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
        return f"Loss function is '{self.name}'"

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

    @property
    def recommended_output_activation(self):
        if self.allowed_activation_functions:
            return self.allowed_activation_functions[0]
        return Activation_NoDamnFunction  # Safe fallback

    def validate_activation_functions(self):
        """
        Ensures the Gladiator is using a valid activation function setup for this loss function.
        - 🚨 Errors if the output activation is incompatible.
        - ⚠️ Warnings if the hidden layer activation is suboptimal.
        """

        output_activation = Neuron.output_neuron.activation
        hidden_activations = [neuron.activation for layer in Neuron.layers[:-1] for neuron in layer]  # All except output layer

        # 🚨 Hard error for output activation mismatch
        if self.allowed_activation_functions and output_activation not in self.allowed_activation_functions:
            raise ValueError(
                f"🚨 Invalid output activation {output_activation} for {self.name}. "
                f"\nAllowed: {', '.join([act.name for act in self.allowed_activation_functions])}"
            )
        #TODO need to warn example like MSE with sig..
        """ below not working for none
        # ⚠️ Warning for hidden activations (optional)
        if self.recommended_hidden_activations:
            for act in hidden_activations:
                if act not in self.recommended_hidden_activations:
                    print(f"⚠️ Warning: Hidden activation {act} is not ideal for {self.name}. "
                          f"Consider {', '.join([a.name for a in self.recommended_hidden_activations])}")
        """


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

Loss_MSE = StrategyLossFunction(
    loss=mse_loss,
    derivative=mse_derivative,
    name="Mean Squared Error (MSE)",
    short_name="MSE",
    desc="Calculates the average of the squares of differences between predictions and actual values.",
    when_to_use="Commonly used for regression problems.",
    best_for="Regression tasks.",
    allowed_activations=None,  # ✅ All activations allowed
    derivative_formula="2 * (prediction - target)",
    bd_defaults= [0, 1, 0.5]
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

Loss_MAE = StrategyLossFunction(
    loss=mae_loss,
    derivative=mae_derivative,
    name="Mean Absolute Error (MAE)",
    short_name="MAE",
    desc="Calculates the average of the absolute differences between predictions and actual values.",
    when_to_use="Useful for regression tasks less sensitive to outliers.",
    best_for="Regression tasks with outlier presence.",
    allowed_activations=None,  # ✅ All activations allowed
    derivative_formula="sign(prediction - target) / n",
    bd_defaults= [0, 1, 0.5]
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
    sigmoid_logits = 1 / (1 + np.exp(-logits))
    loss_gradient = sigmoid_logits - y_true

    #print(f"DEBUG BCEWLL Derivative:")
    #print(f"  Raw Logits: {logits}")
    #print(f"  Sigmoid(logits): {sigmoid_logits}")
    #print(f"  Target (y_true): {y_true}")
    #print(f"  Loss Gradient: {loss_gradient}")
    #print("-" * 50)  # Separator for readability

    return loss_gradient

Loss_BCEWithLogits = StrategyLossFunction(
    loss=bce_with_logits_loss,
    derivative=bce_with_logits_derivative,
    name="Binary Cross-Entropy with Logits",
    short_name="BCE_WL",
    desc="Numerically stable BCE loss using raw logits instead of Sigmoid outputs.",
    when_to_use="Use this instead of BCE when working with raw logits (no Sigmoid activation in the last layer).",
    best_for="Binary classification tasks where Sigmoid is removed from the model's final layer.",
    derivative_formula="sigmoid(logits) - target",
    #allowed_activations=[Activation_NoDamnFunction],
    allowed_activations=None,
    #bd_rules=(0, 1, "Warning: BCEWithLogits is most efficient with {0,1} targets", "Warning: BCEWithLogits is most efficient with a threshold of 0.0"),
    bd_defaults= [0, 1, 0]
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

Loss_BCE = StrategyLossFunction(
    loss=binary_crossentropy_loss,
    derivative=binary_crossentropy_derivative,
    name="Binary Cross-Entropy",
    short_name="BCE",
    desc="Calculates loss for binary classification tasks using cross-entropy.",
    when_to_use="Ideal for binary classification problems.",
    best_for="Binary classification.",
    derivative_formula="- (target / prediction - (1 - target) / (1 - prediction)) / n",
    allowed_activations=[Activation_Sigmoid],
    #bd_rules=(0, 1, "Error: BCE requires targets to be {0,1}", "Error: BCE requires threshold to be 0.5"),
    bd_defaults= [0, 1, 0.5]
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

Loss_CategoricalCrossEntropy = StrategyLossFunction(
    loss=categorical_crossentropy_loss,
    derivative=categorical_crossentropy_derivative,
    name="Categorical Cross-Entropy",
    short_name="CCL",
    desc="Calculates loss for multi-class classification tasks using cross-entropy.",
    when_to_use="Ideal for multi-class classification problems with one-hot encoded targets.",
    best_for="Multi-class classification.",
    derivative_formula="(prediction - target) / n",
    #bd_rules = (None, None, "NEVER", None),
    bd_defaults= [0, 0, 0]
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
    Loosens strict ±1 updates by blending with the raw margin violation.
    """
    margin = 1 - y_true * y_pred
    margin_violation = margin > 0
    grad = np.where(margin_violation, -y_true * margin, 0)  # Use margin instead of hard ±1

    n = _get_n(y_true)
    return grad / n

Loss_Hinge = StrategyLossFunction(
    loss=hinge_loss,
    derivative=hinge_derivative,
    name="Hinge Loss",
    short_name="Hinge",
    desc="Used primarily for maximum-margin classification (e.g., SVMs).",
    when_to_use="Useful for support vector machines and related models.",
    best_for="Binary classification with margin-based methods.",
    derivative_formula="where(1 - target * prediction > 0, -target, 0) / n",
    allowed_activations=[Activation_NoDamnFunction],
    #bd_rules=(-1, 1, "Error: Hinge requires targets to be {-1,1}", "Error: Hinge requires threshold to be 0.0"),
    bd_defaults= [-1, 1, 0]
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

Loss_LogCosh = StrategyLossFunction(
    loss=logcosh_loss,
    derivative=logcosh_derivative,
    name="Log-Cosh Loss",
    short_name="LCL",
    desc="Calculates loss using the logarithm of the hyperbolic cosine of the prediction error.",
    when_to_use="A smooth loss function that is less sensitive to outliers than MSE.",
    best_for="Regression tasks.",
    allowed_activations=[Activation_NoDamnFunction, Activation_Tanh, Activation_ReLU, Activation_LeakyReLU],
    derivative_formula="tanh(prediction - target) / n",
    bd_defaults= [-1, 1, 0]

)


def huber_loss(y_pred, y_true, delta=1.0):
    """
    Computes the Huber loss.
    - Quadratic for |error| ≤ delta, linear beyond.
    """
    error = y_pred - y_true
    is_small = np.abs(error) <= delta
    # ½·error² for small errors; delta·( |error| – ½·delta ) for large errors
    squared = 0.5 * error**2
    linear = delta * (np.abs(error) - 0.5 * delta)
    return np.mean(np.where(is_small, squared, linear))

def huber_derivative(y_pred, y_true, delta=1.0):
    """
    Derivative of the Huber loss wrt predictions.
    """
    n = _get_n(y_true)
    error = y_pred - y_true
    # error for small; delta·sign(error) for large, all divided by n
    grad = np.where(np.abs(error) <= delta, error, delta * np.sign(error))
    return grad / n

Loss_Huber = StrategyLossFunction(
    loss=huber_loss,
    derivative=huber_derivative,
    name="Huber Loss",
    short_name="Huber",
    desc=(
        "Combines MSE and MAE: uses a squared term for small differences "
        "and a linear term for large differences to reduce outlier impact."
    ),
    when_to_use="Regression problems where you want robustness to outliers.",
    best_for="Regression tasks with potential outliers in the data.",
    allowed_activations=None,  # ✅ All activations allowed
    derivative_formula="error if |error| ≤ δ else δ·sign(error)",
    bd_defaults= [-1, 1, 0]
)

def simple_error_loss(y_pred, y_true):
    """
    Returns a simple average absolute error (not used in updates — for reporting only).
    """
    return np.mean(np.abs(y_pred - y_true))


def simple_error_derivative(y_pred, y_true):
    """
    Returns a raw blame signal: prediction - target.
    This matches the expected direction for weight -= lr * blame * input.
    """
    return y_pred - y_true  # <-- The "goofy" way, but matches your update logic


Loss_SimpleError = StrategyLossFunction(
    loss=simple_error_loss,
    derivative=simple_error_derivative,
    name="Simple Error-Based Update",
    short_name="BLAME",
    desc="Uses prediction - target as a direct blame signal. Matches classic update style of (error * input * lr).",
    when_to_use="Great for binary decisions and early regression with raw directional updates.",
    best_for="Classic perceptrons or any intuitive adjustment model.",
    allowed_activations=None,
    derivative_formula="prediction - target",
    #bd_rules=(0, 1, "Binary Decision using raw error", "Use threshold of 0.5"),
    bd_defaults= [-1, 1, 0]
)



def half_wit_loss(y_pred, y_true):
    """
    Computes the same value as MSE — this is a placeholder to match structure.
    """
    return np.mean((y_pred - y_true) ** 2)  # Or replace with np.mean(abs(y_pred - y_true)) if desired

def half_wit_derivative(y_pred, y_true):
    """
    Computes the raw error: (prediction - target) — skipping the 2x and division.
    """
    return (y_pred - y_true)

Loss_HalfWit = StrategyLossFunction(
    loss=half_wit_loss,
    derivative=half_wit_derivative,
    name="Half Wit Error",
    short_name="HalfWit",
    desc="Returns the raw error instead of a true gradient. Half the math, all the charm.",
    when_to_use="When you want an honest answer.",
    best_for="Situations where clarity or interpretability of error is preferred.",
    allowed_activations=None,
    derivative_formula="(prediction - target)",
    bd_defaults= [-1, 1, 0]
)

def schrodinger_loss(y_pred, y_true):
    """
    Schrodinger Loss: Absolutely no idea which direction to go.
    Calculates the average absolute error but discards all sign information.
    """
    return np.mean(np.abs(y_pred - y_true))

def schrodinger_derivative(y_pred, y_true):
    """
    Derivative of Schrodinger Loss — the optimizer is blindfolded.
    Outputs all +1s (because abs() removes direction), regardless of prediction accuracy.
    """
    n = _get_n(y_true)
    return np.ones_like(y_pred) / n  # No idea whether it's over or under — just nudge it forward

Loss_Schrodinger = StrategyLossFunction(
    loss=schrodinger_loss,
    derivative=schrodinger_derivative,
    name="Schrödinger's Loss",
    short_name="TheCat",
    desc="Punishes error but gives no indication which way to go. Not recommended unless you're trolling your optimizer.",
    when_to_use="When you want maximum confusion and minimum utility.  Uhhh, Mundo no remember Mundo's name",
    best_for="Philosophical debates, not machine learning.  Exception: finding your cat",
    allowed_activations=None,
    derivative_formula="1 / n  (for all values — direction is lost to the void)",
    bd_defaults= [-99, -97, -96]
)
