from typing import List
from dataclasses import dataclass

from enum import Enum

@dataclass
class NNA_history:
    run_id: int
    arena: str
    gladiator: str
    accuracy: float
    architecture: list
    problem_type: str
    best_mae: float
    loss_function: str
    hidden_activation: str
    output_activation: str
    weight_initializer: str
    normalization_scheme: str
    seed: int
    learning_rate: float
    epoch_count: int
    convergence_condition: str
    runtime_seconds: int
    final_mae: float
    sample_count: int
    target_min : float                  # either min numeric or count of smaller class
    target_max : float                  # either max numeric or count of larger class
    target_min_label: str               # e.g., "Repay" or "0"
    target_max_label:str                # e.g., "Default" or "1"
    target_mean : float                 # mean of target values (esp useful in regression)
    target_stdev : float                # standard deviation of targets
    notes : str                 # Optional remarks (e.g., 'testing AdamW with tanh glitch patch')
    rerun_config : str

    @classmethod
    #def from_config(cls, learning_rate: float, epoch_count: int, last_error: float, config):
    def from_config(cls, TRI, config ):
        return cls(
            run_id                  = TRI.run_id,
            arena                   = TRI.training_data.arena_name,
            gladiator               = TRI.gladiator,
            accuracy                = TRI.accuracy,
            architecture            = config.architecture,
            best_mae                = TRI.lowest_mae,
            problem_type            = config.training_data.problem_type,
            loss_function           = config.loss_function.name,
            hidden_activation       = config.hidden_activation.name,
            output_activation       = config.output_activation.name,
            weight_initializer      = config.initializer.name,
            normalization_scheme    = "WIP",
            seed                    = TRI.seed,
            learning_rate           = config.learning_rate,
            epoch_count             = TRI.last_epoch,
            convergence_condition   = TRI.converge_cond or "None",
            runtime_seconds         = TRI.time_seconds,
            final_mae               = TRI.mae,
            sample_count            = TRI.training_data.sample_count,
            target_min              = TRI.training_data.target_min,                       # either min numeric or count of smaller class
            target_max              = TRI.training_data.target_max,                       # either max numeric or count of larger class
            target_min_label        = TRI.training_data.target_min_label,                   # e.g., "Repay" or "0"
            target_max_label        = TRI.training_data.target_max_label,                  # e.g., "Default" or "1"
            target_mean             = TRI.training_data.target_mean,                       # mean of target values (esp useful in regression)
            target_stdev            = TRI.training_data.target_stdev,                       # standard deviation of targets
            notes                   = "notes here",                             # Optional remarks (e.g., 'testing AdamW with tanh glitch patch')
            rerun_config            = "coming soon"
        )

    def display(self):
        print("\n🧬 Reproducibility Snapshot1")
        print("────────────────────────────")
        print(f"Arena:             {self.arena}")
        print(f"Gladiator:         {self.gladiator}")
        print(f"Architecture:      {self.architecture}")
        print(f"Problem Type:      {self.problem_type}")
        print(f"Loss Function:     {self.loss_function}")
        print(f"Hidden AF:         {self.hidden_activation}")
        print(f"Output AF:         {self.output_activation}")
        print(f"Weight Init:       {self.weight_initializer}")
        print(f"Data Norm Scheme:  {self.normalization_scheme}")
        print(f"Seed:              {self.seed}")
        print(f"Learning Rate:     {self.learning_rate}")
        print(f"Epochs Run:        {self.epoch_count}")
        print(f"Convergence Rule:  {self.convergence_condition}")


@dataclass
class ModelInfo:
    #model_id: str
    run_id : int
    gladiator: str
    seconds: float
    cvg_condition: str
    full_architecture: List[int]
    problem_type: str


@dataclass
class Iteration:
    run_id: int
    epoch: int
    iteration: int
    inputs: str  # Serialized as JSON
    inputs_unscaled: str  # Serialized as JSON
    target: float
    target_unscaled: float
    prediction: float  # After threshold(step function) is applied but before unscaling is applied
    prediction_unscaled: float #After unscaling is applied
    prediction_raw: float
    loss_function: str
    loss: float
    loss_gradient: float
    # error: float
    accuracy_threshold: float

    @property
    def error(self):
        return float(self.target - self.prediction_raw)

    @property
    def error_unscaled(self):
        return float(self.target_unscaled - self.prediction_unscaled)

    @property
    def absolute_error_unscaled(self) -> float:
        return float(abs(self.error_unscaled))

    @property
    def absolute_error(self) -> float:
        return float(abs(self.error))

    @property
    def squared_error(self) -> float:
        return self.error ** 2

    @property
    def relative_error(self) -> float:
        return abs(self.error / (self.target + 1e-64))

    @property
    def is_true(self) -> int:
        if self.target == 0:
            return self.prediction == 0  # Prevent divide by zero and floating point issues
        if self.target == self.prediction:
            return True  # for binary decision
        return int(self.relative_error <= self.accuracy_threshold)  # for regression

    @property
    def is_false(self) -> int:
        return not self.is_true

    @property
    def is_true_positive(self) -> int:
        return int(self.is_true and self.target != 0)

    @property
    def is_true_negative(self) -> int:
        return int(self.is_true and self.target == 0)

    @property
    def is_false_positive(self) -> int:
        return int(not self.is_true and self.target == 0)

    @property
    def is_false_negative(self) -> int:
        return int(not self.is_true and self.target != 0)


def ez_debug(**kwargs):
    """
    Print debug information for each provided variable.

    For every keyword argument passed in, this function prints:
    1) The variable name
    2) An equal sign
    3) The variable's value
    4) A tab character for separation

    Example:
        a = 1
        b = 2
        c = 3
        ez_debug(a=a, b=b, c=c)
        # Output: a=1    b=2    c=3
    """
    debug_output = ""
    for name, value in kwargs.items():
        debug_output += f"{name}={value}\t"
    print(debug_output)

class RecordLevel(Enum):
    NONE = 0         # No recording — e.g., hyperparameter sweep (LR probe)
    SUMMARY = 1      # Basic stats: accuracy, loss, convergence, etc.
    FULL = 2         # + Iteration history, weight deltas, etc. (NeuroForge playback)
    DEBUG = 3        # + Diagnostics, blame signals, and dev-level traces
