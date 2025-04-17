from enum import IntEnum, auto

from src.engine import Neuron
from src.engine.Neuron import Neuron

class BatchMode(IntEnum):
    SINGLE_SAMPLE           = auto()  # One sample at a time, fixed order
    MINI_BATCH              = auto()  # Mini-batches in fixed order
    FULL_BATCH              = auto()  # All samples per update (no shuffling)
    #MINI_SHUFFLED           = auto()  # Mini-batches with fresh shuffle each epoch
    #MINI_SHUFFLED_STICKY    = auto()  # Mini-batches using same shuffled order each epoch

#TODO ADD Pitfalls

class Optimizer:
    def __init__(self,
            update_function,
            finalizer_function=None,
            name="",
            desc=""
            , when_to_use=""
            , best_for=""
            , pitfalls=""
            , backprop_popup_headers_batch      = None
            , backprop_popup_operators_batch    = None
            , backprop_popup_headers_finalizer  = None
            , backprop_popup_headers_single     = None
            , backprop_popup_operators_single   = None
            ,backprop_popup_operators_finalizer = None
    ):
        """
        🚀 Strategy object representing an optimization algorithm used to update weights in a neural network.

        This class encapsulates all the logic and metadata associated with an optimizer, including:
        - Update and finalize functions for weight adjustment
        - Descriptive fields for GUI rendering
        - Customizable headers and operator symbols for NeuroForge pop-up debugging

        Attributes:
            name (str): Human-readable name of the optimizer (e.g., "Adam", "SGD").
            desc (str): Short description of what the optimizer does.
            when_to_use (str): Summary of scenarios where this optimizer is a strong choice.
            best_for (str): Types of problems or use cases this optimizer is best suited for.
            update_function (Callable): Called each sample to accumulate gradients or deltas.
            finalizer_function (Callable): Called at batch boundaries to apply updates (optional).
            backprop_popup_headers_single (List[str]): Column headers shown when using single-sample training.
            backprop_popup_operators_single (List[str]): Symbols/operators used between terms in the single-sample popup.
            backprop_popup_headers_batch (List[str]): Column headers shown when using batch training.
            backprop_popup_operators_batch (List[str]): Symbols/operators used between terms in the batch popup.
            backprop_popup_headers_finalizer (List[str]):  Column headers shown gathered during FINALIZE step of using batch training.
            backprop_popup_operators_finalizer(List[str]):  Symbols/operators used between terms in the batch finalizer popup.
        """

        self.update                             = update_function #and apply correct function here with parameter for sticky and shuffled
        self.finalizer_function                 = finalizer_function
        self.name                               = name
        self.desc                               = desc
        self.when_to_use                        = when_to_use
        self.best_for                           = best_for
        self.pitfalls                           = "everywhere"
        self._backprop_popup_headers_batch      = backprop_popup_headers_batch
        self._backprop_popup_operators_batch    = backprop_popup_operators_batch
        self._backprop_popup_headers_finalizer  = backprop_popup_headers_finalizer
        self._backprop_popup_operators_finalizer= backprop_popup_operators_finalizer
        self._backprop_popup_headers_single     = backprop_popup_headers_single
        self._backprop_popup_operators_single   = backprop_popup_operators_single
        self._batch_size                        = None

    def configure_optimizer(self, config):
        # Set batch_size based on batch_mode
        if config.batch_mode    == BatchMode.FULL_BATCH:
            config.batch_size   =  config.training_data.sample_count
        elif config.batch_mode  == BatchMode.SINGLE_SAMPLE:
            config.batch_size   =  1
        elif config.batch_mode  == BatchMode.MINI_BATCH:
            if not (isinstance(config.batch_size, int) and config.batch_size > 0):
                raise ValueError("For MINI_BATCH mode, batch_size must be a positive integer.")
            if config.batch_size > config.training_data.sample_count:
                # Fallback: adjust the batch size and log a warning.
                print(f"Warning: mini_batch size ({config.batch_size}) exceeds total sample count ({config.training_data.sample_count}). Falling back to full-batch mode.")
                config.batch_size = config.training_data.sample_count
        else:
            raise ValueError(f"Unsupported batch_mode: {config.batch_mode}")

        # Return the appropriate interface based on batch_size.
        if config.batch_size == 1:
            return self._backprop_popup_headers_single, self._backprop_popup_operators_single, self._backprop_popup_headers_finalizer, self._backprop_popup_operators_finalizer
        else:
            return self._backprop_popup_headers_batch, self._backprop_popup_operators_batch, self._backprop_popup_headers_finalizer, self._backprop_popup_operators_finalizer



def adam_update(neuron, input_vector, blame, t, config, epoch, iteration, batch_id):
    """
    Adam update function for mini-batch accumulation.

    Instead of applying the weight update immediately, this function
    accumulates gradients for each weight. These accumulated gradients
    will be used during the finalize phase to perform a mini-batch update.

    Parameters:
        neuron: The neuron instance. It is expected to have:
            - neuron.learning_rates: list of learning rates for bias/weights.
            - neuron.nid: identifier of the neuron.
            - (Optional) neuron.accumulated_gradients: list of floats.
        input_vector: List or array of inputs corresponding to each weight.
        blame: The error signal for this neuron.
        t: Time step (not used here, kept for interface consistency).
        config: Configuration object containing batch_size and other settings.
        epoch: Current epoch number.
        iteration: Current iteration number.
        batch_id: Identifier for the batch

    Returns:
        logs: A list of log entries for debugging/analysis.

    Note:
        This function assumes that the actual weight update and moving
        average updates (m and v) are performed in the adam_finalize function.
    """
    logs = []

    for weight_id, input in enumerate(input_vector):
        # Compute the per-sample gradient
        raw_adjustment = input * blame
        neuron.accumulated_accepted_blame[weight_id] += raw_adjustment  # Accumulate for batch

        logs.append([epoch, iteration,  neuron.nid, weight_id, batch_id,
             input,                  # Input value
             blame,                  # Blame (error signal)
             raw_adjustment,                   # This sample's contribution: x * blame
             neuron.accumulated_accepted_blame[weight_id]]
        )
    return logs
def adam_finalize(batch_size, epoch, iteration, batch_id):
    """
    Finalize the Adam update for a mini-batch.

    For each neuron, average the accumulated gradients over the batch,
    update the moving averages (m and v) using the averaged gradient,
    and then apply the weight update based on Adam's rule.

    Assumes that:
      - Each neuron has an attribute `accumulated_gradients` (a list of floats),
        which has been incremented per sample.
      - Each neuron has already initialized lists `m` and `v` (one per weight)
        and a time step counter `t` (initialized to 0 if not present).
      - The learning rates for the neuron are stored in `neuron.learning_rates`.

    Parameters:
        batch_size (int): The number of samples accumulated.
        epoch: Current epoch number.
        iteration: Current iteration number.
        batch_id: Identifier for the batch
    Returns:
        logs: A list of log entries for debugging/analysis.
    """
    logs = []
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    # Loop over all neurons
    for layer in Neuron.layers:
        for neuron in layer:
            neuron.t += 1
            # Accumulate update for each weight (and bias at i==0)
            for weight_id in range(len(neuron.accumulated_accepted_blame)):
                # Compute the average gradient for this weight over the mini-batch.
                avg_blame = neuron.accumulated_accepted_blame[weight_id] / batch_size
                                                                                    # Increment the update step counter (per neuron, or you can use a global counter)
                neuron.m[weight_id] = beta1 * neuron.m[weight_id] + (1 - beta1) * avg_blame          # Update the biased first moment estimate
                neuron.v[weight_id] = beta2 * neuron.v[weight_id] + (1 - beta2) * (avg_blame ** 2)   # Update the biased second moment estimate

                # Compute bias-corrected estimates
                m_hat = neuron.m[weight_id] / (1 - beta1 ** neuron.t)
                v_hat = neuron.v[weight_id] / (1 - beta2 ** neuron.t)

                lr = neuron.learning_rates[weight_id]
                adjustment = lr * m_hat / (v_hat ** 0.5 + epsilon)
                #print(f"epoch={epoch} iter={iteration} weight={weight_id}")
                #print(f"  avg_blame={avg_blame:.4e}")
                #print(f"  m={neuron.m[weight_id]:.4e} v={neuron.v[weight_id]:.4e} t={neuron.t}")
                #print(f"  m_hat={m_hat:.4e} v_hat={v_hat:.4e}")
                #print(f"  adjustment={adjustment:.4e}")

                # Apply the final update: assume index 0 corresponds to the bias.
                if weight_id == 0:
                    neuron.bias -= adjustment
                else:
                    neuron.weights[weight_id - 1] -= adjustment
                #logs.append([epoch, iteration,  neuron.nid, weight_id,batch_id,
                logs.append( [epoch, iteration,  neuron.nid, weight_id, batch_id,
                    neuron.accumulated_accepted_blame[weight_id] , batch_size,    #could almost skip if space becomes issue
                    avg_blame,
                    neuron.m[weight_id],
                    neuron.v[weight_id],
                    neuron.t,
                    m_hat,
                    v_hat,
                    #f"adj:{adjustment:.4f}"
                ])
            # Reset accumulated gradients for the next batch.
            neuron.accumulated_accepted_blame = [0.0 for _ in neuron.accumulated_accepted_blame]
    return logs

standard_adam_headers_single        = ["Input", "Blame", "Raw Adj",  "Lrn Rt"]
standard_adam_operators_single      = [  "*",         "=",       "*",       "="]

standard_adam_headers_batch         = ["Input", "Blame", "Raw Adj", "Cum."]
standard_adam_operators_batch       = [ "*",     "=",     " ",          "||"]

standard_adam_headers_finalizer     = ["B Total", "Count", "Avg", "m", "v", "t", "m hat", "v hat"]
standard_adam_operators_finalizer   = [ "/",      "=",     "*", " " , " " , " " , " " , " ", " ", " ", " ", " ", " " ]

################################################################################################################
#################################################################################################################
standard_gbs_headers_single         = ["Input", "Blame", "Raw Adj",  "Lrn Rt"]
standard_gbs_operators_single       = [  "*",         "=",       "*",       "="]

standard_gbs_headers_batch          = ["Input", "Blame", "Raw Adj", "Cum."]
standard_gbs_operators_batch        = [ "*",     "=",     " ",          "||"]

standard_gbs_headers_finalizers     = ["B Total", "Count", "Avg Blm",  "Lrn Rt"]
standard_gbs_operators_finalizers   = ["/",        "=",     "*", " "]


Optimizer_Adam = Optimizer(
    name = "Adam",
    desc = "Adaptive Moment Estimation optimizer with per-weight momentum and scale tracking.",
    when_to_use = "Ideal for handling noisy or sparse gradients; frequently the best default.",
    best_for = "Most deep learning tasks with minimal tuning.",
    update_function = adam_update,
    finalizer_function = adam_finalize,
    backprop_popup_headers_single = standard_adam_headers_single,
    backprop_popup_operators_single = standard_adam_operators_single,
    backprop_popup_headers_batch = standard_adam_headers_batch,
    backprop_popup_operators_batch = standard_adam_operators_batch,
    backprop_popup_headers_finalizer    =standard_adam_headers_finalizer,
    backprop_popup_operators_finalizer  =standard_adam_operators_finalizer
)


def sgd_update(neuron, input_vector, accepted_blame, t, config, epoch, iteration, batch_id):
    """
    Full Batch SGD — accumulate gradients during backprop,
    apply the final adjustment once per batch using average blame.
    This method is called **during each sample**, to accumulate blame.
    standard_gbs_headers =["Input", "Accp Blm", "Raw Adj", "Cum.",  "Batch Tot", "Lrn Rt"]
    """
    logs = []
    #batch_id = (iteration - 1) // config.batch_size #TODO This would be safer as a counter incremented in finalize...
    for weight_id, input_x in enumerate(input_vector):
        raw_adjustment = input_x * accepted_blame
        neuron.accumulated_accepted_blame[weight_id] += raw_adjustment  # Accumulate for batch
        logs.append([epoch, iteration,  neuron.nid, weight_id, batch_id,
                     input_x, accepted_blame, raw_adjustment] +
                    ([neuron.accumulated_accepted_blame[weight_id]] if config.batch_size > 1 else []) #NOTE THIS IS CORRECT TO Use config batch size rather than actual... interface does not change just because it is a leftover.
        )
    return logs


def sgd_finalize(batch_size, epoch, iteration, batch_id):
    """
    Called once per batch to apply the average accumulated blame.
    Resets the accumulation afterward.
    """
    logs = []
    for layer in Neuron.layers:
        for neuron in layer:
            for weight_id, blame_sum in enumerate(neuron.accumulated_accepted_blame):
                avg_blame = blame_sum / batch_size   # Uses actual sample count, not configured batch size — for leftovers at end
                adjustment = neuron.learning_rates[weight_id] * avg_blame
                if weight_id == 0:
                    neuron.bias -= adjustment
                else:
                    neuron.weights[weight_id - 1] -= adjustment
                logs.append([epoch, iteration,  neuron.nid, weight_id,batch_id,
                             blame_sum, batch_size, avg_blame, neuron.learning_rates[weight_id]]
                )

            #clear accumulated accepted blamee for next batch
            neuron.accumulated_accepted_blame = [0.0] * len(neuron.accumulated_accepted_blame)
    return logs

Optimizer_SGD = Optimizer(
    name        = "Stochastic Gradient Descent",
    desc        = "Updates weights using the raw gradient scaled by learning rate.",
    when_to_use = "Simple problems, shallow networks, or when implementing your own optimizer.",
    best_for    = "Manual tuning, simple models, or teaching tools.",
    update_function                     = sgd_update,
    finalizer_function                  = sgd_finalize,
    backprop_popup_headers_batch        = standard_gbs_headers_batch,
    backprop_popup_operators_batch      = standard_gbs_operators_batch,
    backprop_popup_headers_single       = standard_gbs_headers_single,
    backprop_popup_operators_single     = standard_gbs_operators_single,
    backprop_popup_headers_finalizer    = standard_gbs_headers_finalizers,
    backprop_popup_operators_finalizer  = standard_gbs_operators_finalizers
)


"""
def vanilla_GBS_update(neuron, input_vector, blame, t, config, epoch, iteration, gladiator):
    "" "
    SGD update across all weights (including bias).
    input_vector[0] is assumed to be the bias input (usually 1.0).
    "" "
    logs = []

    for i, x in enumerate(input_vector):
        grad = x * blame
        adjustment = neuron.learning_rates[i] * grad

        if i == 0:
            neuron.bias -= adjustment
        else:
            neuron.weights[i - 1] -= adjustment

        #logs.append([
        #    epoch, iteration, gladiator, neuron.nid, i,
        #    x, "*", blame, "*", neuron.learning_rates[i], "=", adjustment
        #])
        logs.append([
            epoch, iteration, gladiator, neuron.nid, i ,  0,
            x, "*",                               # arg1
            blame, "B",                   # arg2
            adjustment, "S",                   # arg3
            neuron.accumulated_accepted_blame[i], " ",  # arg4
            neuron.learning_rates[i], "="                  # arg5
        ])

    return logs

def vanilla_GBS_finalize(batch_size):
    "" "It will be called - but in vanilla update occurs immediately"" "
    pass

Optimizer_Vanilla_GBS = Optimizer(
    update_function=vanilla_GBS_update,
    finalizer_function=vanilla_GBS_finalize,
    name="Vanilla_GBS",
    desc="Old school Gradient Bull Shit.",
    when_to_use="Never - maybe to debug Batch mode",
    best_for="Trying to sound smart",
    backprop_popup_headers_single =standard_gbs_headers_batch,
    backprop_popup_headers_batch =standard_gbs_operators
)
"""
