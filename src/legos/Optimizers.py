from enum import IntEnum, auto

from src.engine import Neuron
from src.engine.Neuron import Neuron
standard_gbs_headers_batch      =["Input", "Blame", "Raw Adj", "Cum.",  "Lrn Rt"]
standard_gbs_headers_single     =["Input", "Blame", "Raw Adj",  "Lrn Rt"]
standard_gbs_operators_batch    =[            "*",         "=",     " ",          "*",       "="]
standard_gbs_operators_single   =[            "*",         "=",       "*",       "="]

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
            , backprop_popup_headers_single     = None
            , backprop_popup_operators_single   = None
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
        self.update = update_function #and apply correct function here with parameter for sticky and shuffled
        self.finalizer_function = finalizer_function
        self.name = name
        self.desc = desc
        self.when_to_use = when_to_use
        self.best_for = best_for
        self.pitfalls = "everywhere"
        self.backprop_popup_headers_batch   = backprop_popup_headers_batch
        self.backprop_popup_operators_batch = backprop_popup_operators_batch
        self.backprop_popup_headers_single   = backprop_popup_headers_single
        self.backprop_popup_operators_single = backprop_popup_operators_single
        self.backprop_popup_headers = None      #single interface for either batch or single
        self.backprop_popup_operators =  None   #single interface for either batch or single
        self.batch_size = None

    def prepare_optimizer(self, config):
        self.batch_size = config.batch_size
        if self.batch_size == 1:
            self.backprop_popup_headers     = self.backprop_popup_headers_single
            self.backprop_popup_operators   =  self.backprop_popup_operators_single #single interface for either
        else:
            self.backprop_popup_headers     = self.backprop_popup_headers_batch #single interface for either
            self.backprop_popup_operators   =  self.backprop_popup_operators_batch #single interface for either



def sgd_update(neuron, input_vector, accepted_blame, t, config, epoch, iteration, gladiator):
    """
    Full Batch SGD — accumulate gradients during backprop,
    apply the final adjustment once per batch using average blame.
    This method is called **during each sample**, to accumulate blame.
    standard_gbs_headers =["Input", "Accp Blm", "Raw Adj", "Cum.",  "Batch Tot", "Lrn Rt"]
    """
    logs = []
    batch_id = (iteration - 1) // config.batch_size
    for weight_id, input_x in enumerate(input_vector):
        raw_adjustment = input_x * accepted_blame
        neuron.accumulated_accepted_blame[weight_id] += raw_adjustment  # Accumulate for batch
        logs.append(
            [epoch, iteration, gladiator, neuron.nid, weight_id, batch_id, input_x, accepted_blame, raw_adjustment] +
            ([neuron.accumulated_accepted_blame[weight_id]] if config.batch_size > 1 else []) +
            [neuron.learning_rates[weight_id]]
        )

    return logs
def sgd_finalize(batch_size):
    """
    Called once per batch to apply the average accumulated blame.
    Resets the accumulation afterward.
    """
    logs = []
    # MOVED TO ENGINE batch_size = config.batch_size  # Full batch
    for layer in Neuron.layers:
        for neuron in layer:
            for i, blame_sum in enumerate(neuron.accumulated_accepted_blame):
                avg_blame = blame_sum / batch_size    #TODO WARNING!!! If there are leftovers, use that not batch size
                adjustment = neuron.learning_rates[i] * avg_blame
                if i == 0:
                    neuron.bias -= adjustment
                else:
                    neuron.weights[i - 1] -= adjustment
            neuron.accumulated_accepted_blame = [0.0] * len(neuron.accumulated_accepted_blame)
    return logs


Optimizer_SGD = Optimizer(
    name        = "Stochastic Gradient Descent",
    desc        = "Updates weights using the raw gradient scaled by learning rate.",
    when_to_use = "Simple problems, shallow networks, or when implementing your own optimizer.",
    best_for    = "Manual tuning, simple models, or teaching tools.",
    update_function                 = sgd_update,
    finalizer_function              = sgd_finalize,
    backprop_popup_headers_batch    = standard_gbs_headers_batch,
    backprop_popup_operators_batch  = standard_gbs_operators_batch,
    backprop_popup_headers_single   = standard_gbs_headers_single,
    backprop_popup_operators_single = standard_gbs_operators_single

)
