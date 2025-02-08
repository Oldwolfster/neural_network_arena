import numpy as np

class LossFunction:
    """
    Encapsulates loss function strategies with optional gradient computation.

    Attributes:
        loss: A function that computes the loss given predictions and true values.
        derivative: A function that computes the gradient of the loss, if available.
        name: The name of the loss function.
        desc: A description of the loss function.
        when_to_use: Guidance on when to use this loss function.
        best_for: The scenarios or tasks where this loss function performs best.
    """
    def __init__(self, loss, derivative=None, name="Custom", desc="", when_to_use="", best_for=""):
        self.loss = loss  # Function to compute the loss.
        self.derivative = derivative  # Optional function to compute the gradient of the loss.
        self.name = name
        self.desc = desc
        self.when_to_use = when_to_use
        self.best_for = best_for

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
    n = y_true.shape[0] if isinstance(y_true, np.ndarray) else len(y_true)
    return 2 * (y_pred - y_true) / n

Loss_MSE = LossFunction(
    loss=mse_loss,
    derivative=mse_derivative,
    name="Mean Squared Error (MSE)",
    desc="Calculates the average of the squares of differences between predictions and actual values.",
    when_to_use="Commonly used for regression problems.",
    best_for="Regression tasks."
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
    n = y_true.shape[0] if isinstance(y_true, np.ndarray) else len(y_true)
    return np.sign(y_pred - y_true) / n

Loss_MAE = LossFunction(
    loss=mae_loss,
    derivative=mae_derivative,
    name="Mean Absolute Error (MAE)",
    desc="Calculates the average of the absolute differences between predictions and actual values.",
    when_to_use="Useful for regression tasks less sensitive to outliers.",
    best_for="Regression tasks with outlier presence."
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
    n = y_true.shape[0] if isinstance(y_true, np.ndarray) else len(y_true)
    return - (y_true / y_pred - (1 - y_true) / (1 - y_pred)) / n

Loss_BinaryCrossEntropy = LossFunction(
    loss=binary_crossentropy_loss,
    derivative=binary_crossentropy_derivative,
    name="Binary Cross-Entropy",
    desc="Calculates loss for binary classification tasks using cross-entropy.",
    when_to_use="Ideal for binary classification problems.",
    best_for="Binary classification."
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
    n = y_true.shape[0]
    return (y_pred - y_true) / n

Loss_CategoricalCrossEntropy = LossFunction(
    loss=categorical_crossentropy_loss,
    derivative=categorical_crossentropy_derivative,
    name="Categorical Cross-Entropy",
    desc="Calculates loss for multi-class classification tasks using cross-entropy.",
    when_to_use="Ideal for multi-class classification problems with one-hot encoded targets.",
    best_for="Multi-class classification."
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
    n = y_true.shape[0] if isinstance(y_true, np.ndarray) else len(y_true)
    return grad / n

Loss_Hinge = LossFunction(
    loss=hinge_loss,
    derivative=hinge_derivative,
    name="Hinge Loss",
    desc="Used primarily for maximum-margin classification (e.g., SVMs).",
    when_to_use="Useful for support vector machines and related models.",
    best_for="Binary classification with margin-based methods."
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
    n = y_true.shape[0] if isinstance(y_true, np.ndarray) else len(y_true)
    return np.tanh(y_pred - y_true) / n

Loss_LogCosh = LossFunction(
    loss=logcosh_loss,
    derivative=logcosh_derivative,
    name="Log-Cosh Loss",
    desc="Calculates loss using the logarithm of the hyperbolic cosine of the prediction error.",
    when_to_use="A smooth loss function that is less sensitive to outliers than MSE.",
    best_for="Regression tasks."
)

"""
# Example usage:
if __name__ == "__main__":
    # --- Regression Examples ---
    y_true_reg = np.array([1.0, 2.0, 3.0])
    y_pred_reg = np.array([1.1, 1.9, 3.2])
    
    print("=== Regression Losses ===")
    print(f"MSE Loss: {Loss_MSE(y_pred_reg, y_true_reg)}")
    print(f"MSE Gradient: {Loss_MSE.grad(y_pred_reg, y_true_reg)}")
    print(f"MAE Loss: {Loss_MAE(y_pred_reg, y_true_reg)}")
    print(f"MAE Gradient: {Loss_MAE.grad(y_pred_reg, y_true_reg)}")
    print(f"Log-Cosh Loss: {Loss_LogCosh(y_pred_reg, y_true_reg)}")
    print(f"Log-Cosh Gradient: {Loss_LogCosh.grad(y_pred_reg, y_true_reg)}")
    
    # --- Binary Classification Example ---
    y_true_bin = np.array([0, 1, 0, 1])
    y_pred_bin = np.array([0.1, 0.9, 0.2, 0.8])
    
    print("\n=== Binary Classification Loss ===")
    print(f"Binary Cross-Entropy Loss: {Loss_BinaryCrossEntropy(y_pred_bin, y_true_bin)}")
    print(f"Binary Cross-Entropy Gradient: {Loss_BinaryCrossEntropy.grad(y_pred_bin, y_true_bin)}")
    
    # --- Multi-class Classification Example ---
    # For demonstration, assume 3 classes and two samples.
    y_true_cat = np.array([[1, 0, 0],
                           [0, 1, 0]])
    y_pred_cat = np.array([[0.7, 0.2, 0.1],
                           [0.1, 0.8, 0.1]])
    
    print("\n=== Multi-class Classification Loss ===")
    print(f"Categorical Cross-Entropy Loss: {Loss_CategoricalCrossEntropy(y_pred_cat, y_true_cat)}")
    print(f"Categorical Cross-Entropy Gradient: {Loss_CategoricalCrossEntropy.grad(y_pred_cat, y_true_cat)}")
"""