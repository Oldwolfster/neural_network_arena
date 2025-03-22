from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import List, Optional, Protocol

# Strategy interfaces
class PropagationStrategy(Protocol):
    def forward_pass(self, sample: np.ndarray, network: 'NeuralNetwork') -> np.ndarray:
        """Execute forward propagation."""
        ...

    def backward_pass(self, sample: np.ndarray, loss_gradient: float, network: 'NeuralNetwork') -> None:
        """Execute backward propagation."""
        ...

class OptimizerStrategy(Protocol):
    def update_weights(self, neuron: 'Neuron', gradient: float) -> None:
        """Update weights based on gradient."""
        ...

class ValidationStrategy(Protocol):
    def validate(self, prediction: float, target: float) -> tuple[float, float, float, float]:
        """Return error, loss, final_prediction, and loss_gradient."""
        ...

# Example implementations
class StandardPropagation:
    def forward_pass(self, sample: np.ndarray, network: 'NeuralNetwork') -> np.ndarray:
        for layer in network.layers:
            for neuron in layer:
                # Your existing forward propagation logic
                pass
        return network.layers[-1][0].activation_value

    def backward_pass(self, sample: np.ndarray, loss_gradient: float, network: 'NeuralNetwork') -> None:
        # Your existing backpropagation logic
        pass

class SGDOptimizer:
    def update_weights(self, neuron: 'Neuron', gradient: float) -> None:
        neuron.weights += neuron.learning_rate * gradient
        neuron.bias += neuron.learning_rate * gradient

class BinaryClassificationValidator:
    def validate(self, prediction: float, target: float) -> tuple[float, float, float, float]:
        error = target - prediction
        loss = 0.5 * error * error  # MSE
        final_prediction = 1 if prediction >= 0.5 else 0
        loss_gradient = error
        return error, loss, final_prediction, loss_gradient

# Updated Gladiator class
@dataclass
class GladiatorConfig:
    gladiator_name: str
    hyperparameters: 'HyperParameters'
    training_data: 'TrainingData'
    ram_db: Optional['Database'] = None

class Gladiator:
    def __init__(
        self,
        config: GladiatorConfig,
        propagation: Optional[PropagationStrategy] = None,
        optimizer: Optional[OptimizerStrategy] = None,
        validator: Optional[ValidationStrategy] = None
    ):
        self.config = config
        self.propagation = propagation or StandardPropagation()
        self.optimizer = optimizer or SGDOptimizer()
        self.validator = validator or BinaryClassificationValidator()

        self.neurons: List[Neuron] = []
        self.layers: List[List[Neuron]] = []  # Move to instance level
        self._initialize_from_config()

    def train(self) -> tuple[str, list[int]]:
        """Main training loop with strategy delegation."""
        if not self.neurons:
            self.initialize_neurons([])

        for epoch in range(self.config.hyperparameters.epochs_to_run):
            convergence_signal = self.run_an_epoch(epoch)
            if convergence_signal:
                return convergence_signal, self._full_architecture

        return "Did not converge", self._full_architecture

    def run_an_epoch(self, epoch_num: int) -> str:
        for i, sample in enumerate(self.training_samples):
            prediction = self.propagation.forward_pass(sample, self)

            error, loss, prediction, loss_gradient = self.validator.validate(
                prediction,
                sample[-1]
            )

            self.propagation.backward_pass(sample, loss_gradient, self)

            # Record iteration data
            self._record_iteration(epoch_num, i, sample, prediction, loss, loss_gradient)

        return self._check_convergence()

# Example custom implementations
class CustomPropagation(StandardPropagation):
    def forward_pass(self, sample: np.ndarray, network: 'NeuralNetwork') -> np.ndarray:
        # Custom implementation
        pass

class AdamOptimizer(SGDOptimizer):
    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Timestep

    def update_weights(self, neuron: 'Neuron', gradient: float) -> None:
        self.t += 1
        if neuron.id not in self.m:
            self.m[neuron.id] = np.zeros_like(neuron.weights)
            self.v[neuron.id] = np.zeros_like(neuron.weights)

        # Update moments
        self.m[neuron.id] = self.beta1 * self.m[neuron.id] + (1 - self.beta1) * gradient
        self.v[neuron.id] = self.beta2 * self.v[neuron.id] + (1 - self.beta2) * gradient * gradient

        # Bias correction
        m_hat = self.m[neuron.id] / (1 - self.beta1 ** self.t)
        v_hat = self.v[neuron.id] / (1 - self.beta2 ** self.t)

        # Update weights
        neuron.weights += neuron.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

# Usage example
config = GladiatorConfig(
    gladiator_name="MyModel",
    hyperparameters=hyper_params,
    training_data=training_data
)

model = Gladiator(
    config=config,
    propagation=CustomPropagation(),
    optimizer=AdamOptimizer(),
    validator=BinaryClassificationValidator()
)
"""
