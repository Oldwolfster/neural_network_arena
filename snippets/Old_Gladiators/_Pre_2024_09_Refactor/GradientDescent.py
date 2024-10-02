from src.ArenaEngine import *
from src.Metrics import Metrics
from src.BaseGladiator import Gladiator

class GradientDescent(Gladiator):
    """
    A simple perceptron implementation for educational purposes.
    This class serves as a template for more complex implementations.

    In order to utilize the metrics there are two steps required:
    1) After each iteration call metrics.record_iteration_metrics
    2) After each iteration call metrics.record_epoch_metrics (no additional info req - uses info from iterations)
    """

    def __init__(self, number_of_epochs: int, metrics: Metrics, *args):
        super().__init__(number_of_epochs, metrics, *args)
        self.bias = 0.0  # Initialize bias term

    def train(self, training_data):
        for epoch in range(self.number_of_epochs):                      # Loop to run specified # of epochs
            if self.run_an_epoch(training_data, epoch):                 # Call function to run single epoch
                break

    def run_an_epoch(self, train_data, epoch_num: int) -> bool:         # Function to run single epoch
        for i, (credit_score, result) in enumerate(train_data):         # Loop through all the training data
            self.training_iteration(i, epoch_num, credit_score, result) # Run single sample of training data
        return self.metrics.record_epoch()                              # Sends back the data for an epoch

    def training_iteration(self, i: int, epoch: int, credit_score: float, result: int) -> None:
        prediction = self.predict(credit_score)                         # Step 1) Guess
        loss = self.compare(prediction, result)                         # Step 2) Check guess, if wrong, how much?
        adjustment_weight, adjustment_bias = self.adjust_parameters(loss, credit_score)  # Step 3) Adjust both weight and bias
        new_weight = self.weight + adjustment_weight
        new_bias = self.bias + adjustment_bias
        self.metrics.record_iteration(i, epoch, credit_score, result, prediction, loss, adjustment_weight, self.weight, new_weight, self.metrics)
        self.weight = new_weight
        self.bias = new_bias

    def predict(self, credit_score: float) -> int:
        return 1 if round(credit_score * self.weight + self.bias, 7) >= 0.5 else 0   # Predict using weight and bias

    def compare(self, prediction: int, result: int) -> float:            # Calculate the Loss
        return result - prediction

    def adjust_parameters(self, loss: float, credit_score: float) -> tuple:  # Adjust both weight and bias
        adjustment_weight = loss * credit_score * self.learning_rate  # Gradient descent for weight
        adjustment_bias = loss * self.learning_rate  # Gradient descent for bias
        return adjustment_weight, adjustment_bias
