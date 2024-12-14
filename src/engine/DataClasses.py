from dataclasses import dataclass


@dataclass
class NetworkFire:
    iteration: int
    epoch: int
    inputs: np.ndarray  # Inputs shared across neurons
    target: float       # Network-level target
    neurons: List[NeuronFire]  # Details for each neuron


@dataclass
class NeuronFire:
    weights: np.ndarray
    new_weights: np.ndarray
    bias: float
    new_bias: float
    activation_output: float  # Output after activation function
