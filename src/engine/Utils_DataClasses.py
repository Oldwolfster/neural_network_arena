from typing import List
from dataclasses import dataclass

from enum import Enum




@dataclass
class ReproducibilitySnapshot:
    arena_name: str
    gladiator_name: str
    architecture: list
    problem_type: str
    best_mae: float
    loss_function_name: str
    hidden_activation_name: str
    output_activation_name: str
    weight_initializer_name: str
    normalization_scheme: str
    seed: int
    learning_rate: float
    epoch_count: int
    convergence_condition: str
    runtime_seconds: int
    final_error: float

    @classmethod
    #def from_config(cls, learning_rate: float, epoch_count: int, last_error: float, config):
    def from_config(cls, config, last_mae: float):
        return cls(
            arena_name=config.training_data.arena_name,
            gladiator_name=config.gladiator_name,
            architecture=config.architecture,
            best_mae= config.lowest_error,
            problem_type=config.training_data.problem_type,
            loss_function_name=config.loss_function.name,
            hidden_activation_name=config.hidden_activation.name,
            output_activation_name=(
                    config.output_activation or config.loss_function.recommended_output_activation
            ).name,
            weight_initializer_name=config.initializer.name,
            normalization_scheme="WIP",
            seed=config.hyper.random_seed,
            learning_rate=config.learning_rate,
            epoch_count=config.final_epoch,
            convergence_condition=config.cvg_condition or "None",
            runtime_seconds=config.seconds,
            final_error=last_mae
        )

    def display(self):
        print("\n🧬 Reproducibility Snapshot1")
        print("────────────────────────────")
        print(f"Arena:             {self.arena_name}")
        print(f"Gladiator:         {self.gladiator_name}")
        print(f"Architecture:      {self.architecture}")
        print(f"Problem Type:      {self.problem_type}")
        print(f"Loss Function:     {self.loss_function_name}")
        print(f"Hidden AF:         {self.hidden_activation_name}")
        print(f"Output AF:         {self.output_activation_name}")
        print(f"Weight Init:       {self.weight_initializer_name}")
        print(f"Data Norm Scheme:  {self.normalization_scheme}")
        print(f"Seed:              {self.seed}")
        print(f"Learning Rate:     {self.learning_rate}")
        print(f"Epochs Run:        {self.epoch_count}")
        print(f"Convergence Rule:  {self.convergence_condition}")


@dataclass
class ModelInfo:
    model_id: str
    seconds: float
    cvg_condition: str
    full_architecture: List[int]
    problem_type: str


@dataclass
class Iteration:
    model_id: str
    epoch: int
    iteration: int
    inputs: str  # Serialized as JSON
    inputs_unscaled: str  # Serialized as JSON
    target: float
    target_unscaled: float
    prediction: float  # After threshold(step function) is applied AND After unscaling is applied
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
