class Optimizer:
    #TODO ADD Pitfalls
    def __init__(self, update_function, name="", desc="", when_to_use="", best_for=""):
        """
        🚀 Encapsulates optimizer strategies.

        Attributes:
            update_function: Function with signature (neuron, i, grad, t, config) -> None
            name: Human-readable name of the optimizer.
            desc: Description of how it works.
            when_to_use: When this optimizer is a good choice.
            best_for: The kinds of problems this optimizer excels at.
        """
        self.update = update_function
        self.name = name
        self.desc = desc
        self.when_to_use = when_to_use
        self.best_for = best_for


def update_sgd(neuron, input_vector, blame, t, config, epoch, iteration, gladiator):
    """
    SGD update across all weights (including bias).
    input_vector[0] is assumed to be the bias input (usually 1.0).
    """
    logs = []

    for i, x in enumerate(input_vector):
        grad = x * blame
        adjustment = neuron.learning_rates[i] * grad

        if i == 0:
            neuron.bias -= adjustment
        else:
            neuron.weights[i - 1] -= adjustment

        logs.append([
            epoch, iteration, gladiator, neuron.nid, i,
            x, "*", blame, "*", neuron.learning_rates[i], "=", adjustment
        ])

    return logs



Optimizer_SGD = Optimizer(
    update_function=update_sgd,
    name="Stochastic Gradient Descent",
    desc="Updates weights using the raw gradient scaled by learning rate.",
    when_to_use="Simple problems, shallow networks, or when implementing your own optimizer.",
    best_for="Manual tuning, simple models, or teaching tools."
)

def update_adam(neuron, input_vector, blame, t, config, epoch, iteration, gladiator):
    """
    Adam optimizer: combines momentum and RMS scaling with bias correction.
    Returns detailed logs for each weight update.
    """
    logs = []
    beta1 = .9 #config.optimizer_beta1  # e.g. 0.9
    beta2 = .999 #config.optimizer_beta2  # e.g. 0.999
    epsilon = 1e-8# config.optimizer_epsilon  # e.g. 1e-8

    for i, x in enumerate(input_vector):
        grad = x * blame
        lr = neuron.learning_rates[i]

        # Update biased first moment estimate
        neuron.m[i] = beta1 * neuron.m[i] + (1 - beta1) * grad
        # Update biased second raw moment estimate
        neuron.v[i] = beta2 * neuron.v[i] + (1 - beta2) * (grad ** 2)

        # Compute bias-corrected moments
        m_hat = neuron.m[i] / (1 - beta1 ** t)
        v_hat = neuron.v[i] / (1 - beta2 ** t)

        # Final weight adjustment
        adjustment = lr * m_hat / (v_hat**0.5 + epsilon)
        #print(f"grad: {grad:.6f}, m: {neuron.m[i]:.6f}, v: {neuron.v[i]:.6f}, m̂: {m_hat:.6f}, v̂: {v_hat:.6f}, adj: {adjustment:.6f}")

        # Apply the update
        if i == 0:
            neuron.bias -= adjustment
        else:
            neuron.weights[i - 1] -= adjustment

        logs.append([
            epoch, iteration, gladiator, neuron.nid, i,
            x, "*", blame, "*", lr, "=", adjustment
        ])

    return logs



Optimizer_Adam = Optimizer(
    update_function=update_adam,
    name="Adam",
    desc="Adaptive Moment Estimation optimizer with per-weight momentum and scale tracking.",
    when_to_use="Ideal for noisy gradients or sparse data. Frequently the best default.",
    best_for="Most deep learning tasks with minimal tuning."
)

def update_adabelief(neuron, input_vector, blame, t, config, epoch, iteration, gladiator):
    """
    AdaBelief optimizer: like Adam, but replaces the second moment estimate with
    the variance of the gradient from the prediction (Belief about the gradient).
    Returns detailed logs for each weight update.
    """
    logs = []
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    for i, x in enumerate(input_vector):
        grad = x * blame
        lr = neuron.learning_rates[i]

        # First moment (momentum)
        neuron.m[i] = beta1 * neuron.m[i] + (1 - beta1) * grad

        # Second moment: squared deviation from belief
        grad_diff = grad - neuron.m[i]
        neuron.v[i] = beta2 * neuron.v[i] + (1 - beta2) * (grad_diff ** 2)

        # Bias correction
        m_hat = neuron.m[i] / (1 - beta1 ** t)
        v_hat = neuron.v[i] / (1 - beta2 ** t)

        adjustment = lr * m_hat / ((v_hat ** 0.5) + epsilon)

        if i == 0:
            neuron.bias -= adjustment
        else:
            neuron.weights[i - 1] -= adjustment

        logs.append([
            epoch, iteration, gladiator, neuron.nid, i,
            x, "*", blame, "*", lr, "=", adjustment
        ])

    return logs

Optimizer_AdaBelief = Optimizer(
    update_function=update_adabelief,
    name="AdaBelief",
    desc="Like Adam, but trusts gradients that match expectation and dampens surprises.",
    when_to_use="Great when convergence stability is more important than sheer speed.",
    best_for="Complex regression tasks, chaotic datasets."
)
