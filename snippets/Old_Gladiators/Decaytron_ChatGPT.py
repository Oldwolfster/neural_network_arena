from src.engine.Metrics import GladiatorOutput
from src.gladiators.BaseGladiator import Gladiator
import random

class Decaytron_ChatGPT(Gladiator):
    """
    A perceptron with weight decay to prevent overfitting and smarter weight initialization based on input variance.
    """

    def __init__(self, number_of_epochs: int, metrics_mgr: MetricsMgr, *args):
        super().__init__(number_of_epochs, metrics_mgr, *args)
        self.weight = random.uniform(-0.1, 0.1)  # Initialize weight randomly in a small range
        self.bias = 0.0  # Introducing bias term
        self.decay_factor = 0.0001  # Weight decay factor
        self.learning_rate = 0.1  # Feel free to adjust this

    def training_iteration(self, training_data) -> GladiatorOutput:
        credit_score = training_data[0]
        result = training_data[1]

        prediction = self.predict(credit_score)  # Step 1: Make a prediction
        loss = self.compare(prediction, result)  # Step 2: Calculate the error (loss)
        adjustment = self.adjust_weight(loss)    # Step 3: Adjust weight based on error

        # Weight decay - penalize larger weights to avoid overfitting
        adjusted_weight = (self.weight + adjustment) * (1 - self.decay_factor)
        new_bias = self.bias + (self.learning_rate * loss * 0.01)

        gladiator_output = GladiatorOutput(
            prediction=prediction,
            adjustment=adjustment,
            weight=self.weight,
            new_weight=adjusted_weight,
            bias=self.bias,
            new_bias=new_bias
        )

        # Apply weight and bias updates
        self.weight = adjusted_weight
        self.bias = new_bias

        return gladiator_output

    def predict(self, credit_score: float) -> int:
        return 1 if round(credit_score * self.weight + self.bias, 7) >= 0.5 else 0

    def compare(self, prediction: float, result: float) -> float:
        return result - prediction

    def adjust_weight(self, loss: float) -> float:
        return loss * self.learning_rate
