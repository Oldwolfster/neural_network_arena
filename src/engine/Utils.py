from dataclasses import dataclass, field
import numpy as np


def chunk_list(lst: list, chunk_size: int):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


import numpy as np

def smart_format(number):
    # Define the formatting for a single number
    def format_single(num):
        if num == 0:
            return "0"
        elif abs(num) < .001:
            return f"{num:,.6f}"
        elif abs(num) < 1:
            return f"{num:,.3f}"
        elif abs(num) > 1000:
            return f"{num:,.0f}"
        else:
            return f"{num:,.2f}"

    # Check if `number` is an array or iterable, format each element if so
    if isinstance(number, (np.ndarray, list, tuple)):
        # Apply `format_single` to each element in the array or list
        vectorized_format = np.vectorize(format_single)
        return vectorized_format(number)
    else:
        # If it's a single number, just format it
        return format_single(number)



def smart_formatOld(number):
    if number == 0:
        return "0"
    elif abs(number) < 1:
        # For very small numbers, show 4 decimal places
        return f"{number:,.4f}"
    elif abs(number) > 1000:
        # For large numbers, show no decimal places
        return f"{number:,.0f}"
    else:
        # For moderate numbers, show 2 decimal places
        return f"{number:,.3f}"




def determine_problem_type(data):
    """
    Examine training data to determine if it's binary decision or regression
    """
    # Extract unique values from the second element of each tuple
    unique_values = set(item[1] for item in data)
    #print(f"determine{unique_values}")
    # If there are only two unique values, it's likely a binary decision problem
    if len(unique_values) == 2:
        return "Binary Decision"

    # If there are more than two unique values, it's likely a regression problem
    elif len(unique_values) > 2:
        return "Regression"

    # If there's only one unique value or the list is empty, it's inconclusive
    else:
        return "Inconclusive"


@dataclass
class GladiatorOutput:
    prediction: float
    #adjustment: float

@dataclass
class IterationContext:
    iteration: int
    epoch: int
    # Old from when it was only 1 input: float
    inputs: np.ndarray
    weights: np.ndarray
    new_weights: np.ndarray
    target: float
    bias: float = 0
    new_bias: float = 0


@dataclass
class GladiatorOutputOrig:
    prediction: float
    adjustment: float
    weight: np.ndarray
    new_weight: np.ndarray
    bias: float = 0
    new_bias: float = 0

@dataclass
class IterationContextOrig:
    iteration: int
    epoch: int
    # Old from when it was only 1 input: float
    inputs: np.ndarray
    target: float

@dataclass
class IterationResult:
    gladiator_output: GladiatorOutput
    context: IterationContext


@dataclass
class EpochSummary:
    # Stored once per epoch
    model_name = ""
    epoch: int = 0
    final_weight = 0
    final_bias = 0
    total_samples: int = 0  #Could this be removed?
    # Accumulated over epoch
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    total_absolute_error: float = 0.0
    total_squared_error: float = 0.0
    total_error: float = 0.0


    # calculated values
    correct: int = 0
    wrong: int = 0
    accuracy: float = field(init=False)
    precision: float = field(init=False)
    recall: float = field(init=False)
    f1: float = field(init=False)
    mean_absolute_error: float = field(init=False)
    mean_squared_error: float = field(init=False)
    rmse: float = field(init=False)
    r_squared: float = field(init=False)
    log_loss: float = field(init=False)
    mape: float = field(init=False)

    # Values i don't think i need
    sum_target: float = 0.0
    sum_prediction: float = 0.0
    sum_target_squared: float = 0.0
    sum_prediction_squared: float = 0.0
    sum_target_prediction: float = 0.0
    sum_mape: float = 0.0
