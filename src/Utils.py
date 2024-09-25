from dataclasses import dataclass, field


def smart_format(number):
    if abs(number) < 1:
        # For very small numbers, show 4 decimal places
        return f"{number:,.4f}"
    elif abs(number) > 1000:
        # For large numbers, show no decimal places
        return f"{number:,.0f}"
    else:
        # For moderate numbers, show 2 decimal places
        return f"{number:,.2f}"


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
    model_name: str
    epoch: int
    total_samples: int
    correct_predictions: int
    total_absolute_error: float
    total_squared_error: float

