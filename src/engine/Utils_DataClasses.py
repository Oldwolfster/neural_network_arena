from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class ModelInfo:
    model_id: str
    seconds: float
    cvg_condition: str
    full_architecture: List[int]

@dataclass
class Iteration:
    model_id: str
    epoch: int
    iteration: int
    inputs: str  # Serialized as JSON
    target: float
    prediction: float
    loss: float
    # error: float
    accuracy_threshold : float

    @property
    def error(self):
        if isinstance(self.prediction, (list, np.ndarray)):
            # Handle multi-output case (e.g., sum of squared errors)
            return float(np.sum((np.array(self.target) - np.array(self.prediction)) ** 2))
        else:
            # Single-output case
            return float(self.target - self.prediction)


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
        if self.accuracy_threshold == -696:
            raise ValueError("accuracy_threshold has not been set!")
        return int(self.relative_error <= self.accuracy_threshold)

    @property
    def is_false(self) -> int:
        if self.accuracy_threshold == -696:
            raise ValueError("accuracy_threshold has not been set!")
        return int(self.relative_error > self.accuracy_threshold)

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
