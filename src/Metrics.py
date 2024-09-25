from Utils import *
from typing import List


class Metrics:
    # Class variable (shared by all instances)
    accuracy_threshold = -696  # Default threshold, will fail loudly if not set

    def __init__(self, data: IterationResult):
        self.epoch          = data.context.epoch
        self.iteration      = data.context.iteration
        self.input          = data.context.input
        self.target         = data.context.target
        self.prediction     = data.gladiator_output.prediction
        self.weight         = data.gladiator_output.weight
        self.new_weight     = data.gladiator_output.new_weight
        self.bias           = data.gladiator_output.bias
        self.new_bias       = data.gladiator_output.new_bias

    @classmethod
    def set_acc_threshold(cls, threshold: float):
        cls.accuracy_threshold = threshold  # Allows updating the shared accuracy threshold

    @property
    def error(self) -> float:
        return self.target - self.prediction

    @property
    def absolute_error(self) -> float:
        return abs(self.error)

    @property
    def squared_error(self) -> float:
        return self.error ** 2


    @property
    def relative_error(self) -> float:
        return abs(self.error / (self.target + 1e-64))

    @property
    def is_correct(self) -> bool:
        if self.accuracy_threshold == -696:
            raise ValueError("accuracy_threshold has not been set!")
        return self.relative_error <= self.accuracy_threshold

    @property
    def is_true_positive(self) -> bool:
        return self.is_correct and self.target != 0

    @property
    def is_true_negative(self) -> bool:
        return self.is_correct and self.target == 0

    @property
    def is_false_positive(self) -> bool:
        return not self.is_correct and self.target == 0

    @property
    def is_false_negative(self) -> bool:
        return not self.is_correct and self.target != 0

    def to_list(self) -> List:
        return [
            self.epoch,
            self.iteration,
            self.input,
            self.target,
            self.prediction,
            self.error,
            self.absolute_error, #6
            self.squared_error,
            self.relative_error,
            self.is_correct,
            self.weight,   #10
            self.new_weight,
            self.bias,
            self.new_bias
        ]