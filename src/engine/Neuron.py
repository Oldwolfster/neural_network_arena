import numpy as np

from src.engine.ActivationFunction import *
from src.engine.RamDB import RamDB


class Neuron:
    """
    Represents a single neuron with weights, bias, and an activation function.
    """
    layers = []  # Shared across all Gladiators, needs resetting per run
    def __init__(self, nid: int, num_of_weights: int, learning_rate: float, weight_initializer, layer_id: int = 0, activation = None):
        #print(f"creating neuron - nid={nid}")
        self.nid = nid
        self.layer_id = layer_id  # Add layer_id to identify which layer the neuron belongs to
        self.num_of_weights = num_of_weights
        self.learning_rate = learning_rate
        self.raw_sum = 0.0
        self.activation_value = 0.0
        self.activation =  activation or Linear         # function
        self.activation_gradient = 0.0  # Store activation gradient from forward pass
        self.error_signal = 1111.11
        self.weight_adjustments = ""
        self.error_signal_calcs = ""

        # ✅ Apply weight initializer strategy
        #self.weight_initializer = weight_initializer  # Store the strategy
        self.initialize_weights(weight_initializer)
        self.neuron_inputs = np.zeros_like(self.weights)

        # ✅ Ensure activation is never None
        self.activation = activation if activation is not None else Linear
        self.activation_name = self.activation.name  # ✅ No more AttributeError

        # Ensure layers list is large enough to accommodate this layer_id
        while len(Neuron.layers) <= layer_id:
            Neuron.layers.append([])

        # Add neuron to the appropriate layer and set its position
        Neuron.layers[layer_id].append(self)
        self.position = len(Neuron.layers[layer_id]) - 1  # Zero-based position within layer


    def initialize_weights(self, weight_initializer):

        self.weights, self.bias = weight_initializer((self.num_of_weights,))
        self.weights_before = self.weights.copy()
        self.bias_before = self.bias

    def reinitialize(self, new_initializer):
        """Reinitialize weights & bias using a different strategy."""
        #self.weight_initializer = new_initializer
        self.initialize_weights(new_initializer)

    def activate(self):
        """Applies the activation function."""
        self.activation_value = self.activation(self.raw_sum)
        self.activation_gradient = self.activation.apply_derivative(self.activation_value)  # Store gradient!

    def set_activation(self, activation_function):
        """Dynamically update the activation function."""
        self.activation = activation_function
        self.activation_name = activation_function.name

    def compute_gradient(self):
        """Use the derivative for backpropagation."""
        return self.activation.apply_derivative(self.activation_value)
    @classmethod
    def reset_layers(cls):
        """ Clears layers before starting a new Gladiator. """
        cls.layers = []


    @staticmethod
    def bulk_insert_weights(db, model_id, epoch, iteration):
        """
        Collects all weight values across neurons and creates a bulk insert SQL statement.
        """
        sql_statements = []

        # Ensure model_id is wrapped in single quotes if it's a string
        model_id_str = f"'{model_id}'" if isinstance(model_id, str) else model_id

        for layer in Neuron.layers:
            for neuron in layer:
                for weight_id, (prev_weight, weight ) in enumerate(zip(neuron.weights_before, neuron.weights)):
                    sql_statements.append(
                        f"({model_id_str}, {epoch}, {iteration}, {neuron.nid}, {weight_id + 1}, {prev_weight}, {weight})"
                    )

                # Store bias separately as weight_id = 0
                sql_statements.append(
                    f"({model_id_str}, {epoch}, {iteration}, {neuron.nid}, 0, {neuron.bias_before}, {neuron.bias})"
                )

        if sql_statements:
            sql_query = f"INSERT INTO Weights (model_id, epoch, iteration, nid, weight_id, value_before, value) VALUES {', '.join(sql_statements)};"
            #print(f"Query for weights: {sql_query}")
            db.execute(sql_query)


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

import numpy as np





# 1. Uniform Random Initialization (between -1 and 1)
def initialize_uniform_random(neurons):
    for neuron in neurons:
        neuron.weights = np.random.uniform(-1, 1, size=len(neuron.weights))
        neuron.bias = np.random.uniform(-1, 1)

# 2. Normal Distribution Initialization (mean=0, std=1)
def initialize_normal_random(neurons):
    for neuron in neurons:
        neuron.weights = np.random.normal(0, 1, size=len(neuron.weights))
        neuron.bias = np.random.normal(0, 1)

# 3. Xavier/Glorot Initialization (good for sigmoid/tanh activations)
def initialize_xavier(neurons):
    for neuron in neurons:
        limit = np.sqrt(6 / (len(neuron.weights) + 1))  # +1 for bias
        neuron.weights = np.random.uniform(-limit, limit, size=len(neuron.weights))
        neuron.bias = np.random.uniform(-limit, limit)

# 4. He Initialization (good for ReLU activations)
def initialize_he(neurons):
    for neuron in neurons:
        limit = np.sqrt(2 / len(neuron.weights))
        neuron.weights = np.random.normal(0, limit, size=len(neuron.weights))
        neuron.bias = np.random.normal(0, limit)

# 5. Small Random Values (close to zero)
def initialize_small_random(neurons):
    for neuron in neurons:
        neuron.weights = np.random.randn(len(neuron.weights)) * 0.01
        neuron.bias = np.random.randn() * 0.01
