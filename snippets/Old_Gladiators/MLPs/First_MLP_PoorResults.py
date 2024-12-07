from src.engine.Metrics import GladiatorOutput
from src.gladiators.BaseGladiator import Gladiator
import numpy as np

class First_MLP_ChatGPT(Gladiator):
    """
    A Multi-Layer Perceptron with two hidden layers and ReLU activation functions.
    Designed to explore overfitting potential on noisy data.
    """

    def __init__(self, number_of_epochs: int, metrics_mgr: MetricsMgr, *args):
        super().__init__(number_of_epochs, metrics_mgr, *args)

        # MLP with 2 hidden layers (5 neurons in each layer)
        self.input_size = 1    # Single input (credit score)
        self.hidden_size1 = 5  # First hidden layer size
        self.hidden_size2 = 5  # Second hidden layer size
        self.output_size = 1   # Single output (binary decision)

        # Initialize weights randomly for all layers
        self.weights1 = np.random.randn(self.input_size, self.hidden_size1)
        self.weights2 = np.random.randn(self.hidden_size1, self.hidden_size2)
        self.weights_output = np.random.randn(self.hidden_size2, self.output_size)

        # Bias terms for each layer
        self.bias1 = np.zeros((1, self.hidden_size1))
        self.bias2 = np.zeros((1, self.hidden_size2))
        self.bias_output = np.zeros((1, self.output_size))

    def training_iteration(self, training_data) -> GladiatorOutput:
        credit_score = training_data[0]
        result = training_data[1]

        # Forward pass
        input_data = np.array([[credit_score]])  # Convert input to 2D array
        hidden_layer1 = self.relu(np.dot(input_data, self.weights1) + self.bias1)
        hidden_layer2 = self.relu(np.dot(hidden_layer1, self.weights2) + self.bias2)
        prediction = self.sigmoid(np.dot(hidden_layer2, self.weights_output) + self.bias_output)

        # Loss calculation (Binary cross-entropy loss)
        loss = result - prediction[0][0]

        # Backward pass (Simple gradient descent updates)
        adjustment_output = loss * prediction * (1 - prediction)
        adjustment_hidden2 = np.dot(adjustment_output, self.weights_output.T) * self.relu_derivative(hidden_layer2)
        adjustment_hidden1 = np.dot(adjustment_hidden2, self.weights2.T) * self.relu_derivative(hidden_layer1)

        # Update weights and biases
        self.weights_output += np.dot(hidden_layer2.T, adjustment_output) * self.learning_rate
        self.bias_output += adjustment_output * self.learning_rate

        self.weights2 += np.dot(hidden_layer1.T, adjustment_hidden2) * self.learning_rate
        self.bias2 += adjustment_hidden2 * self.learning_rate

        self.weights1 += np.dot(input_data.T, adjustment_hidden1) * self.learning_rate
        self.bias1 += adjustment_hidden1 * self.learning_rate

        # For simplicity, we return the first weight as the representative weight
        gladiator_output = GladiatorOutput(
            prediction=int(round(prediction[0][0])),  # Binary decision
            adjustment=adjustment_output[0][0],
            weight=self.weights1[0][0],  # Using the first weight from the input layer for display
            new_weight=self.weights1[0][0] + adjustment_hidden1[0][0],  # Updated weight
            bias=self.bias_output[0][0],  # Output bias
            new_bias=self.bias_output[0][0] + adjustment_output[0][0]  # Updated output bias
        )

        return gladiator_output

    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return np.where(x > 0, 1, 0)

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
