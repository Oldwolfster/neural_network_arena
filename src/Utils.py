from dataclasses import dataclass, field


def chunk_list(lst: list, chunk_size: int):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def smart_format(number):
    if number == 0:
        return "0"
    elif abs(number) < 1:
        # For very small numbers, show 4 decimal places
        return f"{number:,.6f}"
    elif abs(number) > 1000:
        # For large numbers, show no decimal places
        return f"{number:,.0f}"
    else:
        # For moderate numbers, show 2 decimal places
        return f"{number:,.2f}"


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
    adjustment: float
    weight: float
    new_weight: float
    bias: float = 0
    new_bias: float = 0

@dataclass
class IterationContext:
    iteration: int
    epoch: int
    input: float
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
