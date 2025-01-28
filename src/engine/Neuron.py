import numpy as np


class Neuron:
    """
    Represents a single neuron with weights, bias, and an activation function.
    """
    def __init__(self, nid: int, num_of_weights: int, learning_rate: float, layer_id: int = 0):
        #print(f"creating neuron - nid={nid}")
        self.nid = nid
        self.layer_id = layer_id  # Add layer_id to identify which layer the neuron belongs to
        self.num_of_weights = num_of_weights
        self.weights = np.array([(nid + 1) * 10 for _ in range(num_of_weights)], dtype=np.float64)
        self.neuron_inputs = np.zeros_like(self.weights)
        self.weights_before = self.weights.copy()       #Done so it's available to create view
        self.bias = float(nid +1)                      # Explicitly set as float
        self.bias_before = self.bias                    #Done so it's available to create view
        self.learning_rate = learning_rate
        self.raw_sum = 0.0
        self.activation_value = 0.0
        self.activation = "Linear"
        self.output = 0 #TODO need to populate this in child model???
        #print(f"INSTANTIATE NEURON nid={self.nid}\tself.num_of_weights{self.num_of_weights}")

        #Coming soon self.activation_function = activation_function




    """
    the below methods restrict experimenting to much.
    def forward(self, inputs: np.ndarray) -> float:
        "" "
        Compute the neuron's output given the inputs.
        "" "
        weighted_sum = np.dot(self.weights, inputs) + self.bias
        return self.activation_function(weighted_sum)

    def update(self, inputs: np.ndarray, error: float):
        "" "
        Update the weights and bias based on the error.
        "" "
        self.weights += self.learning_rate * error * inputs
        self.bias += self.learning_rate * error
"""