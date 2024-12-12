import numpy as np


class Neuron:
    """
    Represents a single neuron with weights, bias, and an activation function.
    """
    def __init__(self, nid: int, input_count: int,  learning_rate: float): #activation_function: callable,
        self.input_count = input_count
        self.weights = np.full(self.input_count, 0) #TODO will change 0 to nid after matching single NN
        self.learning_rate = learning_rate
        self.bias = 0
        #self.weights = np.random.uniform(-1.0, 1.0, size=input_count)
        #self.bias = np.random.uniform(-1.0, 1.0)

        #Coming soon self.activation_function = activation_function
        #Coming soonself.learning_rate = learning_rate

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