from dataclasses import dataclass

import numpy as np


@dataclass
class NetworkFire:
    epoch: int
    iteration: int
    inputs: np.ndarray  # Inputs shared across neurons
    target: float       # Network-level target
    #This is already in the Gladiator class.  i don't think we need t move itneurons: List[NeuronFire]  # Details for each neuron


@dataclass
class NeuronFire:
    epoch: int
    iteration: int
    nid: int
    weights: np.ndarray
    new_weights: np.ndarray
    bias: float
    new_bias: float
    activation_output: float  # Output after activation function
