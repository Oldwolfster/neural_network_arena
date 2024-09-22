import numpy as np
from typing import List, Tuple
from src.TrainingPit import *
from src.Metrics import Metrics
from src.Gladiator import Gladiator


class Simpletron_Gradient_Descent_Claude(Gladiator):
    """
    A Simpletron implementation using gradient descent for binary classification of linearly separable data.
    """

    def __init__(self, number_of_epochs: int, metrics: Metrics, learning_rate: float = 0.01):
        """
        Initialize the SimpletronGradientDescent model.

        Args:
            number_of_epochs (int): The maximum number of training epochs.
            metrics (Metrics): An object to track and record model metrics.
            learning_rate (float): The learning rate for weight updates. Defaults to 0.01.
        """
        super().__init__(number_of_epochs)
        self.metrics = metrics
        self.weight = np.random.randn()
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def train(self, training_data: List[Tuple[float, int]]) -> None:
        """
        Train the model on the provided data.

        Args:
            training_data (List[Tuple[float, int]]): A list of tuples, where each tuple contains
                                                     a feature value (float) and a label (int, 0 or 1).
        """
        for epoch in range(self.number_of_epochs):
            if self.run_an_epoch(training_data, epoch):
                break

    def run_an_epoch(self, train_data: List[Tuple[float, int]], epoch_num: int) -> bool:
        """
        Run a single epoch of training.

        Args:
            train_data (List[Tuple[float, int]]): The training data.
            epoch_num (int): The current epoch number.

        Returns:
            bool: True if the model has converged, False otherwise.
        """
        total_loss = 0
        for i, (feature, label) in enumerate(train_data):
            total_loss += self.training_iteration(i, epoch_num, feature, label)

        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch_num}, Average Loss: {avg_loss}")
        return self.metrics.record_epoch()

    def training_iteration(self, i: int, epoch: int, feature: float, label: int) -> float:
        """
        Perform a single training iteration.

        Args:
            i (int): The iteration number within the epoch.
            epoch (int): The current epoch number.
            feature (float): The input feature value.
            label (int): The expected output (0 or 1).

        Returns:
            float: The computed loss for this iteration.
        """
        prediction = self.predict_prob(feature)
        loss = self.compute_loss(prediction, label)
        self.adjust_parameters(feature, label, prediction)
        self.metrics.record_iteration(i, epoch, feature, label, prediction, loss, 0, self.weight, self.weight,
                                      self.metrics)
        return loss

    def predict_prob(self, feature: float) -> float:
        """
        Make a probability prediction based on the input feature.

        Args:
            feature (float): The input feature value.

        Returns:
            float: The predicted probability (between 0 and 1).
        """
        z = feature * self.weight + self.bias
        return 1 / (1 + np.exp(-z))

    def predict(self, feature: float) -> int:
        """
        Make a binary prediction based on the input feature.

        Args:
            feature (float): The input feature value.

        Returns:
            int: The predicted label (0 or 1).
        """
        return 1 if self.predict_prob(feature) >= 0.5 else 0

    def compute_loss(self, prediction: float, label: int) -> float:
        """
        Compute the binary cross-entropy loss.

        Args:
            prediction (float): The predicted probability.
            label (int): The actual label (0 or 1).

        Returns:
            float: The computed loss.
        """
        # epsilon = 1e-15  # Small value to avoid log(0)
        # return -label * np.log(prediction + epsilon) - (1 - label) * np.log(1 - prediction + epsilon)
        return label - prediction

    def adjust_parameters(self, feature: float, label: int, prediction: float) -> None:
        """
        Adjust the weight and bias using gradient descent.

        Args:
            feature (float): The input feature value.
            label (int): The actual label (0 or 1).
            prediction (float): The predicted probability.
        """
        d_loss = prediction - label
        self.weight -= self.learning_rate * d_loss * feature
        self.bias -= self.learning_rate * d_loss

    def evaluate(self, test_data: List[Tuple[float, int]]) -> float:
        """
        Evaluate the model on test data.

        Args:
            test_data (List[Tuple[float, int]]): The test data.

        Returns:
            float: The accuracy of the model on the test data.
        """
        correct = sum(1 for feature, label in test_data if self.predict(feature) == label)
        return correct / len(test_data)