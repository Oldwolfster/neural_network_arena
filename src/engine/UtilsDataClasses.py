from typing import Any, List
import numpy as np
from dataclasses import dataclass

@dataclass
class IterationData:
    epoch: int
    iteration: int
    inputs: np.ndarray  # Expecting a numpy array
    target: float
    prediction: float
    error: float
    loss: float

    def to_list(self) -> List[Any]:
        return [
            self.epoch,
            self.iteration,
            self.inputs.tolist(),  # Serialize numpy array to list
            self.target,
            self.prediction,
            self.error,
            self.loss
        ]

@dataclass
class NeuronData:
    neuron_id: int
    inputs: np.ndarray  # Expecting a numpy array
    weights: np.ndarray  # Numpy array
    bias: float
    new_weights: np.ndarray  # Numpy array
    new_bias: float
    activation_function: str
    output: float

    def to_list(self) -> List[Any]:
        return [
            self.neuron_id,
            self.inputs.tolist(),  # Serialize numpy array to list
            self.weights.tolist(),  # Serialize numpy array to list
            self.bias,
            self.new_weights.tolist(),  # Serialize numpy array to list
            self.new_bias,
            self.activation_function,
            self.output
        ]
