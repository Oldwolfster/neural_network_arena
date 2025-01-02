import numpy as np


class Neuron:
    """
    Represents a single neuron with weights, bias, and an activation function.
    """
    def __init__(self, nid: int, input_count: int, learning_rate: float, layer_id: int = 0):
        #print(f"creating neuron - nid={nid}")
        self.nid = nid+1
        self.layer_id = layer_id  # Add layer_id to identify which layer the neuron belongs to
        self.input_count = input_count


        self.weights = np.array([(nid + 1) * 10 for _ in range(input_count)], dtype=np.float64)
        self.weights_before = self.weights.copy()       #Done so it's available to create view
        self.bias = float(nid * 0)                      # Explicitly set as float
        self.bias_before = self.bias                    #Done so it's available to create view
        self.learning_rate = learning_rate
        self.activation = "Linear"
        self.output = 0 #TODO need to populate this in child model???

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