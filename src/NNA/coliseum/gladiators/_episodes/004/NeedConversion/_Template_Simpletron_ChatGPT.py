from src.NNA.engine.Metrics import GladiatorOutput
from src.NNA.engine.BaseGladiator import Gladiator

class _Template_Simpletron_ChatGPT(Gladiator):
    """
    An adaptive perceptron implementation to improve convergence.
    It dynamically adjusts the learning rate based on the magnitude of the error.
    """

    def __init__(self, number_of_epochs: int, metrics_mgr: MetricsMgr, *args):
        super().__init__(number_of_epochs, metrics_mgr, *args)
        self.bias = 0.0  # Introduce a bias term for better tuning

    def training_iteration(self, training_data) -> GladiatorOutput:
        credit_score = training_data[0]
        result = training_data[1]
        prediction  = self.predict(credit_score)                        # Step 1) Guess
        loss        = self.compare(prediction, result)                  # Step 2) Check guess, if wrong, how much?

        # Adaptive learning rate: The larger the loss, the bigger the adjustment
        adjusted_learning_rate = self.learning_rate * (1 + abs(loss))

        adjustment  = self.adjust_weight(loss, adjusted_learning_rate)  # Step 3) Adjust weight dynamically
        new_weight  = self.weight + adjustment                          # Step 3) Adjust(Apply)
        new_bias    = self.bias + (adjusted_learning_rate * loss * 0.01)  # Tiny bias adjustment

        gladiator_output = GladiatorOutput(
            prediction=prediction,
            adjustment=adjustment,
            weight=self.weight,
            new_weight=new_weight,
            bias=self.bias,
            new_bias=new_bias
        )

        # Update the model weights and bias
        self.weight = new_weight
        self.bias = new_bias

        return gladiator_output

    def predict(self, credit_score: float) -> int:
        # Adjusted prediction with bias term
        return 1 if round(credit_score * self.weight + self.bias, 7) >= 0.5 else 0

    def compare(self, prediction: float, result: float) -> float:            # Calculate the Loss
        return result - prediction

    def adjust_weight(self, loss: float, adjusted_learning_rate: float) -> float:
        # Apply dynamic learning rate to the weight adjustment
        return loss * adjusted_learning_rate
