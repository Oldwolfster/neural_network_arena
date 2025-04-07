from enum import IntEnum, auto

from src.engine import Neuron
from src.engine.Neuron import Neuron



class BatchMode(IntEnum):
    SINGLE_ORDERED           = auto()  # One sample at a time, fixed order
    SINGLE_SHUFFLED          = auto()  # One sample at a time, reshuffle each epoch
    SINGLE_SHUFFLED_STICKY   = auto()  # One sample at a time, shuffled once and reused
    MINI_ORDERED             = auto()  # Mini-batches in fixed order
    MINI_SHUFFLED            = auto()  # Mini-batches with fresh shuffle each epoch
    MINI_SHUFFLED_STICKY     = auto()  # Mini-batches using same shuffled order each epoch
    FULL_BATCH               = auto()  # All samples per update (no shuffling)


#TODO ADD Pitfalls

class Optimizer:
    def __init__(self,
            update_function,
            finalizer_function=None,
            batch_mode= None,
            name="",
            desc=""
            , when_to_use=""
            , best_for=""
            , pitfalls=""
            , backprop_popup_headers=None
    ):
        """
        🚀 Encapsulates optimizer strategies.

        Attributes:
            update_function: Function with signature (neuron, i, grad, t, config) -> None
            name: Human-readable name of the optimizer.
            desc: Description of how it works.
            when_to_use: When this optimizer is a good choice.
            best_for: The kinds of problems this optimizer excels at.
            pitfalls:  Everywhere
            backprop_popup_headers = Column headers in popup.  Defaults to SGD style but can override in optimizers like headers.
        """
        if backprop_popup_headers is None:
            backprop_popup_headers = ["", ""]
        self.update = update_function #and apply correct function here with parameter for sticky and shuffled

        self.finalizer_function = finalizer_function
        self.batch_mode = batch_mode
        self.name = name
        self.desc = desc
        self.when_to_use = when_to_use
        self.best_for = best_for
        self.pitfalls = "everywhere"
        if backprop_popup_headers is not None:
            self.backprop_popup_headers = backprop_popup_headers


    """
    def assign_optimizer_functions(optimizer: Optimizer):
        base_name = optimizer.name.lower().replace(" ", "_")
        for mode in ["single", "batch_mini", "batch_full"]:
            func_name = f"update_{base_name}_{mode}"
            if func := globals().get(func_name):
                setattr(optimizer, f"update_function_{mode}", func)
    """

def update_sgd(neuron, input_vector, blame, t, config, epoch, iteration, gladiator):
    if config.batch_mode == BatchMode.SINGLE_ORDERED:
        return update_sgd_single_ordered(neuron, input_vector, blame, t, config, epoch, iteration, gladiator)
    elif config.batch_mode >= BatchMode.MINI_ORDERED:
        return update_sgd_full_batch(neuron, input_vector, blame, t, config, epoch, iteration, gladiator)
    else:
        raise ValueError(f"Optimizer 'Gradient Descent' does not support mode: ({config.batch_mode})")


def update_sgd_single_ordered(neuron, input_vector, blame, t, config, epoch, iteration, gladiator):
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

def update_sgd_full_batch(neuron, input_vector, blame, t, config, epoch, iteration, gladiator):
    """
    Full Batch SGD — accumulate gradients during backprop,
    apply the final adjustment once per batch using average blame.
    This method is called **during each sample**, to accumulate blame.
    """
    logs = []
    for i, x in enumerate(input_vector):
        grad = x * blame
        neuron.accumulated_accepted_blame[i] += grad  # Accumulate for batch

        logs.append([
            epoch, iteration, gladiator, neuron.nid, i,
            x, "*", blame, "*", grad, "=", neuron.accumulated_accepted_blame[i]
        ])
    return  logs

def finalize_sgd_full_batch(config, epoch, gladiator):
    """
    Called once per batch to apply the average accumulated blame.
    Resets the accumulation afterward.
    """
    logs = []
    batch_size = config.training_data.sample_count #Full batch

    for layer in Neuron.layers:
        for neuron in layer:
            for i, grad_sum in enumerate(neuron.accumulated_accepted_blame):
                avg_grad = grad_sum / batch_size
                adjustment = neuron.learning_rates[i] * avg_grad

                if i == 0:
                    neuron.bias -= adjustment
                else:
                    neuron.weights[i - 1] -= adjustment

        #logs.append([
        #    epoch, iteration, gladiator, neuron.nid, i,
        #    "Σblame", grad_sum, "/", batch_size,
        #    "LR", neuron.learning_rates[i], "=", adjustment
        #])

        # Clear accumulated blame for next batch
        neuron.accumulated_accepted_blame = [0.0] * len(neuron.accumulated_accepted_blame)
    return logs


Optimizer_SGD = Optimizer(
    update_function=update_sgd,
    finalizer_function=finalize_sgd_full_batch,
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
    #beta1 = .9 #config.optimizer_beta1  # e.g. 0.9
    #beta2 = .999 #config.optimizer_beta2  # e.g. 0.999
    beta1 = .5 #config.optimizer_beta1  # e.g. 0.9
    beta2 = .9 #config.optimizer_beta2  # e.g. 0.999
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

        """
        Field	Meaning
        x * blame	The raw gradient (before any smoothing or scaling)
        m, v	Momentum and RMS accumulators (biased)
        t	The current timestep (can be per-weight or global, you're transitioning to per-weight)
        m̂, v̂	Bias-corrected versions of m and v
        lr	Your base learning rate
        adj	Final adjustment applied to the weight or bias
        """
        logs.append([
            #below two work...
            #epoch, iteration, gladiator, neuron.nid, i,
            #x, "*", blame, "*", lr, "=", adjustment



            epoch, iteration, gladiator, neuron.nid, i,
            #f"grad: {x:.3f}*{blame:.3f}={grad:.3f}",
            "XX",
            f"m:{neuron.m[i]:.3f}",
            f"v:{neuron.v[i]:.3f}",
            f"t:{t}",
            f"mh:{m_hat:.3f}",
            f"vh:{v_hat:.3f}",
            "",  # <-- Pad for visual grouping

            # for the moment, lets put this creaee an if for Optimizer_Adam vs everything else...
            # sql = """
            #    INSERT INTO WeightAdjustments
            #    (epoch, iteration, model_id, nid, weight_index, arg_1, op_1, arg_2, op_2, arg_3, op_3, result)
            # VALUES (?, ?, ?, ?, ?, CAST(? AS REAL), ?, CAST(? AS REAL), ?, CAST(? AS REAL), ?, CAST(? AS REAL))"""
            #
        ])
        #for reference
        #x, "*", blame, "=", grad,
        #"m", neuron.m[i], "v", neuron.v[i], "t", t,
        #"m̂", m_hat, "v̂", v_hat,

    return logs



Optimizer_Adam = Optimizer(
    update_function=update_adam,
    name="Adam",
    desc="Adaptive Moment Estimation optimizer with per-weight momentum and scale tracking.",
    when_to_use="Ideal for noisy gradients or sparse data. Frequently the best default.",
    best_for="Most deep learning tasks with minimal tuning.",
    backprop_popup_headers=["t", "m", "v", "m̂", "v̂", "Adj"]
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
