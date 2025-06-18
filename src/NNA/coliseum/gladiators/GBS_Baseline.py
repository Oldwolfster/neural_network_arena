import math
import torch
from typing import Tuple
from src.NNA.engine.BaseGladiator import Gladiator
from src.NNA.engine.Neuron import Neuron
from src.NNA.engine.convergence.ConvergenceDetector import ROI_Mode
from src.Legos.LossFunctions import Loss_MSE
from src.Legos.WeightInitializers import Initializer_Xavier
from src.Legos.ActivationFunctions import *


class GBS_The_GPU_Friendly(Gladiator):
    """
    🚀 GBS_The_GPU_Friendly:
    A high-performance gradient-based model using batched tensor math
    ✅ GPU-compatible
    ✅ Vectorized weight updates
    ✅ Clean, non-tinker-friendly (but blazing fast!)
    """

    def config_options(self, config) -> None:
        self.learning_rate = 0.001
        config.loss_function = Loss_MSE
        config.roi_mode = ROI_Mode.SWEET_SPOT
        config.training_data.set_normalization_min_max()

    def initialize(self, config) -> None:
        self.initialize_neurons(
            architecture=[2, 3, 1],
            initializers=[Initializer_Xavier],
            hidden_activation=Activation_Tanh,
        )
        Neuron.output_neuron.set_activation(Activation_NoDamnFunction)
    def forward_pass(self, training_sample: Tuple[float, float, float]) -> None:
        inputs = torch.tensor(training_sample[:-1], dtype=torch.float32)

        for layer in Neuron.layers:
            outputs = []
            for neuron in layer:
                print(f"neuron bias\t{neuron.bias}")
                print(f"neuron weights\t{neuron.weights}")
                input_tensor = torch.cat([torch.tensor([1.0]), inputs])  # Add bias term
                neuron.input_tensor = input_tensor
                weight_tensor = torch.tensor([neuron.bias] + list(neuron.weights), dtype=torch.float32)
                print(f"input tensor\t{input_tensor}")
                print(f"weight tensor\t{weight_tensor}")

                z = torch.dot(input_tensor, weight_tensor)
                neuron.raw_sum = z.item()
                neuron.activate()
                outputs.append(neuron.activation_value)
            inputs = torch.tensor(outputs, dtype=torch.float32)  # Becomes input to next layer

    def back_pass(self, training_sample: Tuple[float, float, float], loss_gradient: float) -> None:
        target = training_sample[-1]

        # --- BACKPROPAGATION: Compute deltas ---
        # Process layers in reverse order.
        # Assume Neuron.layers is a list of layers, where each layer is a list of neurons.

        # Output layer: for each neuron, delta = loss_gradient * activation_gradient
        output_layer = Neuron.layers[-1]
        for neuron in output_layer:
            # Here loss_gradient is dL/da, and neuron.activation_gradient is da/dz.
            neuron.delta = loss_gradient * neuron.activation_gradient

        # Hidden layers: for each neuron, delta = (sum over next layer of (weight * delta)) * activation_gradient.
        # Iterate from second-to-last layer down to the first layer.
        for layer_index in range(len(Neuron.layers) - 2, -1, -1):
            current_layer = Neuron.layers[layer_index]
            next_layer = Neuron.layers[layer_index + 1]
            for j, neuron in enumerate(current_layer):
                weighted_delta_sum = 0.0
                # For each neuron in the next layer, add contribution from the weight connecting current neuron j.
                for next_neuron in next_layer:
                    # next_neuron.weights[j] is the weight from current neuron j to the next neuron.
                    weighted_delta_sum += next_neuron.weights[j] * next_neuron.delta
                neuron.delta = weighted_delta_sum * neuron.activation_gradient

        # --- WEIGHT AND BIAS UPDATE ---
        # Now update each neuron’s weights and bias using the stored input_tensor.
        # (Make sure your forward_pass assigns: neuron.input_tensor = input_tensor)
        for layer in Neuron.layers:
            for neuron in layer:
                # The input_tensor was built as: [1.0] concatenated with the inputs.
                # So the first element (index 0) corresponds to the bias input (always 1.0).
                # Compute gradient for bias (delta * 1.0) and update:
                bias_grad = neuron.delta * neuron.input_tensor[0]
                neuron.bias = neuron.bias - self.learning_rate * bias_grad

                # Update each weight.
                # The i-th weight corresponds to the (i+1)-th element in neuron.input_tensor.
                for i in range(len(neuron.weights)):
                    weight_grad = neuron.delta * neuron.input_tensor[i + 1]
                    neuron.weights[i] = neuron.weights[i] - self.learning_rate * weight_grad


    def back_passMyoldfriend(self, training_sample: Tuple[float, float, float], loss_gradient: float) -> None:
        target = training_sample[-1]

        # 1️⃣ Output layer blame
        self.back_pass__determine_blame_for_output_neuron(loss_gradient)

        # 2️⃣ Hidden layer blame
        for layer_idx in reversed(range(len(self.neurons) - 1)):
            for neuron in self.neurons[layer_idx]:
                self.back_pass__determine_blame_for_a_hidden_neuron(neuron)
        """
        # 3️⃣ Tensorized weight update
        for layer_idx, layer in enumerate(self.neurons):
            if layer_idx == 0:
                prev_values = torch.tensor([1.0] + list(training_sample[:-1]), dtype=torch.float32)  # Include bias
            else:
                prev_activations = [n.activation_value for n in self.neurons[layer_idx - 1]]
                prev_values = torch.tensor([1.0] + prev_activations, dtype=torch.float32)
        """

        # 3️⃣ Tensorized weight update
        for layer_idx, layer in enumerate(self.neurons):
            if layer_idx == 0:
                prev_values = torch.tensor([1.0] + list(training_sample[:-1]), dtype=torch.float32)  # Include bias
            else:
                prev_activations = [n.activation_value for n in self.neurons[layer_idx - 1]]
                prev_values = torch.tensor([1.0] + prev_activations, dtype=torch.float32)

            for neuron in layer:
                blame = neuron.error_signal
                weight_tensor = torch.tensor([neuron.bias] + neuron.weights, dtype=torch.float32)
                grad = blame * prev_values
                updated_weights = weight_tensor - self.learning_rate * grad

                # Reassign
                neuron.bias = updated_weights[0].item()
                neuron.weights = updated_weights[1:].tolist()
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  BELOW HERE IS ALL BOILERPLATE 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  RECOMMEND TO CUSTOMIZE ABOVE! 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    def __init__(self, config):
        self.config_options(config)
        super().__init__(config)
        self.initialize(config)