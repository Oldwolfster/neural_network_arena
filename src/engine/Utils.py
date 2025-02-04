from dataclasses import dataclass, field
import numpy as np
import inspect

import pygame


def chunk_list(lst: list, chunk_size: int):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]




def print_call_stack():
    stack = inspect.stack()
    for frame in stack:
        print(f"Function: {frame.function}, Line: {frame.lineno}, File: {frame.filename}")

def smart_format(num):
    if num == 0:
        return "0"
    elif abs(num) < 1e-6:  # Use scientific notation for very small numbers
        return f"{num:.2e}"
    elif abs(num) < 0.001:  # Use 6 decimal places for small numbers
        formatted = f"{num:,.6f}"
    elif abs(num) < 1:  # Use 3 decimal places for numbers less than 1
        formatted = f"{num:,.3f}"
    elif abs(num) > 1000:  # Use no decimal places for large numbers
        formatted = f"{num:,.0f}"
    else:  # Default to 2 decimal places
        formatted = f"{num:,.2f}"

    # Remove trailing zeros and trailing decimal point if necessary
    return formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted

def store_num(number):
    formatted = f"{number:,.6f}"
    return formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted

def smart_format_Feb03(number):
    # Define the formatting for a single number
    def format_single(num):
        if num == 0:
            return "0"
        elif abs(num) < .001:
            return f"{num:.1e}"
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

def draw_gradient_rect( surface, rect, color1, color2):
    for i in range(rect.height):
        ratio = i / rect.height
        blended_color = [
            int(color1[j] * (1 - ratio) + color2[j] * ratio) for j in range(3)
        ]
        pygame.draw.line(surface, blended_color, (rect.x, rect.y + i), (rect.x + rect.width, rect.y + i))





"""DELETE ME 
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
    #new_iteration_data : IterationData
    #new_neuron_list: list[NeuronData]

from dataclasses import dataclass, field
import math

@dataclass
class EpochSummary:
    model_name: str = ""
    epoch: int = 0
    final_weight: float = 0
    final_bias: float = 0
    total_samples: int = 0

    # error metrics
    total_absolute_error: float = 0.0
    total_squared_error: float = 0.0  #TODO Convert to calculatedd
    total_error: float = 0.0

    # Confusion matrix
    tp: int = 0  # True Positives
    tn: int = 0  # True Negatives
    fp: int = 0  # False Positives
    fn: int = 0  # False Negatives

    @property
    def precision(self) -> float: # Precision (Positive Predictive Value)  #self.precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0

    @property
    def recall(self) -> float: # Recall (Sensitivity)   self.recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0

    @property
    def F1(self) -> float:    # F1 Score
        return (2 * self.precision * self.recall) / (self.precision + self.recall) \
                if (self.precision + self.recall) > 0 else 0
    @property
    def correct(self) -> int:
        return self.tp + self.tn

    @property
    def wrong(self) -> int:
        return self.fp + self.fn

    @property
    def accuracy(self) -> float:
        return self.correct / self.total_samples * 100 if self.total_samples > 0 else 0

    @property
    def mean_absolute_error(self) -> float:
        return self.total_absolute_error / self.total_samples if self.total_samples > 0 else 0

    @property
    def mean_squared_error(self) -> float:
        return self.total_squared_error / self.total_samples if self.total_samples > 0 else 0

    @property
    def rmse(self) -> float:
        return math.sqrt(self.mean_squared_error) if self.mean_squared_error > 0 else 0



        # R-squared (this would typically require predicted and actual values,
        # so this is a placeholder calculation)
        self.r_squared = 1 - (self.total_squared_error /
                               (self.total_samples * math.pow(self.mean_absolute_error, 2))) \
            if self.total_samples > 0 and self.mean_absolute_error != 0 else 0

        # Log Loss (this is a simplified version and might need more context)
        # Assumes binary classification
        self.log_loss = -(self.tp * math.log(self.precision) +
                          self.fn * math.log(1 - self.precision)) / self.total_samples \
            if self.total_samples > 0 else 0

        # Mean Absolute Percentage Error
        self.mape = (self.total_absolute_error / self.total_samples) * 100 \
            if self.total_samples > 0 else 0
"""