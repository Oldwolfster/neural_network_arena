from typing import List, Tuple
from src.BaseArena import *
from src.engine.Metrics import Metrics
from src.gladiators.BaseGladiator import Gladiator

class Template_Simpletron_Claude(Gladiator):
    """
    The simplest implementation of a neural network for the purpose of education.
    This class serves as a template for more complex implementations.
    """

    def __init__(self, number_of_epochs: int, metrics: Metrics, learning_rate: float = 0.001, initial_weight: float = 0.2):
        """
        Initialize the Simpletron model.

        Args:
            number_of_epochs (int): The maximum number of training epochs.
            metrics (Metrics): An object to track and record model metrics.
            learning_rate (float): The learning rate for weight updates. Defaults to 0.001.
            initial_weight (float): The initial weight value. Defaults to 0.2.
        """
        super().__init__(number_of_epochs)
        self.metrics = metrics
        self.weight = initial_weight
        self.learning_rate = learning_rate

    def train(self, training_data):
        """
        Train the model on the provided data.

        Args:
            training_data (List[Tuple[float, int]]): A list of inputs and results
        """
        for epoch in range(self.number_of_epochs):
            if self.run_an_epoch(training_data, epoch):
                break

    def run_an_epoch(self, train_data, epoch_num: int) -> bool:
        """
        Run a single epoch of training.

        Args:
            train_data (List[Tuple[float, int]]): The training data.
            epoch_num (int): The current epoch number.

        Returns:
            bool: True if the model has converged, False otherwise.
        """
        for i, (credit_score, result) in enumerate(train_data):
            self.training_iteration(i, epoch_num, credit_score, result)
        return self.metrics.record_epoch()

    def training_iteration(self, i: int, epoch: int, credit_score: float, result: int) -> None:
        """
        Perform a single training iteration.

        Args:
            i (int): The iteration number within the epoch.
            epoch (int): The current epoch number.
            credit_score (float): The input credit score.
            result (int): The expected output (0 or 1).
        """
        prediction = self.predict(credit_score)
        loss = self.compare(prediction, result)
        adjustment = self.adjust_weight(loss, credit_score)
        new_weight = self.weight + adjustment
        self.metrics.record_iteration(i, epoch, credit_score, result, prediction, loss, adjustment, self.weight, new_weight, self.metrics)
        self.weight = new_weight

    def predict(self, credit_score: float) -> int:
        """
        Make a prediction based on the input.

        Args:
            credit_score (float): The input.

        Returns:
            int: The predicted result (0 or 1).
        """
        return 1 if round(credit_score * self.weight, 7) >= 0.5 else 0

    def compare(self, prediction: int, result: int) -> float:
        """
        Compare the prediction to the actual result to compute the loss.

        Args:
            prediction (int): The predicted value (0 or 1).
            result (int): The actual value (0 or 1).

        Returns:
            float: The computed loss.
        """
        return result - prediction

    def adjust_weight(self, loss: float, credit_score: float) -> float:
        """
        Compute the weight adjustment based on the loss and input.

        Args:
            loss (float): The computed loss.
            credit_score (float): The input credit score.

        Returns:
            float: The weight adjustment.
        """
        return loss * self.learning_rate