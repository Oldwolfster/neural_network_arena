from src.engine.Metrics import GladiatorOutput
from src.gladiators.BaseGladiator import Gladiator
import numpy as np

class MLP_Gladiator(Gladiator):
    """
    A simple Multi-Layer Perceptron with one hidden layer.
    This class is designed to potentially overfit the training data.
    """

    def __init__(self, number_of_epochs: int, metrics_mgr: MetricsMgr, *args):
        super().__init__(number_of_epochs, metrics_mgr, *args)
        self.learning_rate = 0.01
        np.random.seed(42)  # For reproducibility

        # Initialize weights and biases
        self.input_size = 1
        self.hidden_size = 10  # Number of neurons in the hidden layer
        self.output_size = 1

        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def training_iteration(self, training_data) -> GladiatorOutput:
        X = np.array([[training_data[0]]]) / 100.0  # Normalize input to [0, 1]
        y = np.array([[training_data[1]]])

        # Forward pass
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.sigmoid(z1)  # Hidden layer activation
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)  # Output layer activation

        prediction = 1 if a2 >= 0.5 else 0

        # Compute loss (Binary Cross-Entropy)
        loss = y - a2

        # Backward pass
        d_a2 = loss     #gradient of the loss with respect to the output a2.
        d_z2 = d_a2 * self.sigmoid_derivative(a2)
        d_W2 = np.dot(a1.T, d_z2)
        d_b2 = d_z2

        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * self.sigmoid_derivative(a1)
        d_W1 = np.dot(X.T, d_z1)
        d_b1 = d_z1

        # Update weights and biases
        self.W2 += self.learning_rate * d_W2
        self.b2 += self.learning_rate * d_b2
        self.W1 += self.learning_rate * d_W1
        self.b1 += self.learning_rate * d_b1

        # For GladiatorOutput, we can use the mean of weights and biases
        gladiator_output = GladiatorOutput(
            prediction=prediction,
            adjustment=float(np.mean(loss)),
            weight=float(np.mean(self.W2)),
            new_weight=float(np.mean(self.W2)),
            bias=float(np.mean(self.b2)),
            new_bias=float(np.mean(self.b2))
        )

        return gladiator_output

    def predict(self, credit_score: float) -> int:
        X = np.array([[credit_score]]) / 100.0  # Normalize input to [0, 1]
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        return 1 if a2 >= 0.5 else 0

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(a):
        return a * (1 - a)
