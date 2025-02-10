from dataclasses import dataclass
from typing import List

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from src.engine.ActivationFunction import *
from src.engine.WeightInitializer import *



@dataclass
class ModelInfo:
    model_id: str
    seconds: float
    cvg_condition: str
    full_architecture: List[int]
    problem_type : str

@dataclass
class Iteration:
    model_id: str
    epoch: int
    iteration: int
    inputs: str  # Serialized as JSON
    target: float
    prediction: float   #After threshold(step function) is applied
    prediction_raw: float
    loss_function: str
    loss: float
    loss_gradient: float
    # error: float
    accuracy_threshold : float

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
            return self.prediction == 0     #Prevent divide by zero and floating point issues
        if self.target == self.prediction:
            return True                     #for binary decision
        return int(self.relative_error <= self.accuracy_threshold)

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
