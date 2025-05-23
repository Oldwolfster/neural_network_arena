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
