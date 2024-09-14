from src.Arena import *
from src.Metrics import Metrics
from src.Gladiator import Gladiator

class _Template_Simpletron(Gladiator):
    """
    A simple perceptron implementation for educational purposes.
    This class serves as a template for more complex implementations.

    Leverage metrics with just two steps required:
    1) After each iteration call metrics.record_iteration_metrics
    2) After each iteration call metrics.record_epoch_metrics (no additional info req - uses info from iterations)
    """

    def __init__(self, number_of_epochs: int, metrics: Metrics, *args):
        super().__init__(number_of_epochs, metrics, *args)
        # Ideally avoid overriding these, but specific models may need to; so must be free to do so  # It keeps comparisons straight if respected
        # self.weight = override_weight                 # Default  set in parent class
        # self.learning_rate = override_learning_rate   # Default  set in parent class

    def train(self, training_data):
        for epoch in range(self.number_of_epochs):                      # Loop to run specified # of epochs
            if self.run_an_epoch(training_data, epoch):                 # Call function to run single epoch
                break

    def run_an_epoch(self, train_data, epoch_num: int) -> bool:         # Function to run single epoch
        for i, (credit_score, result) in enumerate(train_data):         # Loop through all the training data
            self.training_iteration(i, epoch_num, credit_score, result) # Run single sample of training data
        return self.metrics.record_epoch()                              # Sends back the data for an epoch

    def training_iteration(self, i: int, epoch: int, credit_score: float, result: int) -> None:
        prediction  = self.predict(credit_score)                        # Step 1) Guess
        loss        = self.compare(prediction, result)                  # Step 2) Check guess, if wrong, how much?
        adjustment  = self.adjust_weight(loss)                          # Step 3) Adjust(Calc)
        new_weight  = self.weight + adjustment                          # Step 3) Adjust(Apply)
        self.metrics.record_iteration(i, epoch, credit_score, result, prediction, loss, adjustment, self.weight, new_weight, self.metrics)
        self.weight = new_weight

    def predict(self, credit_score: float) -> int:
        return 1 if round(credit_score * self.weight, 7) >= 0.5 else 0   # Rounded to 7 decimals to avoid FP errors

    def compare(self, prediction: int, result: int) -> float:            # Calculate the Loss
        return result - prediction

    def adjust_weight(self, loss: float) -> float:                       # Apply learning rate to loss.
        return loss * self.learning_rate
