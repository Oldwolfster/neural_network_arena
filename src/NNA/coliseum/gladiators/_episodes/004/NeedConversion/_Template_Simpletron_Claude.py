from src.NNA.engine.Metrics import GladiatorOutput
from src.NNA.engine.BaseGladiator import Gladiator


class _Template_Simpletron_Claude(Gladiator):
    """
    A simple perceptron implementation for educational purposes.
    This class serves as a template for more complex implementations.

    The training data is a tuple where the first element is the input and the second is the target
    After processing return a GladiatorOutput object with the fields populated. Neural Network Arena will handle the rest
    """

    def __init__(self, number_of_epochs: int, metrics_mgr: MetricsMgr, *args):
        super().__init__(number_of_epochs, metrics_mgr, *args)
        self.bias = 0.0  # Initialize bias

    def training_iteration(self, training_data) -> GladiatorOutput:
        credit_score = training_data[0]
        result = training_data[1]
        prediction = self.predict(credit_score)                        # Step 1) Guess
        loss = self.compare(prediction, result)                        # Step 2) Check guess, if wrong, how much?
        weight_adjustment, bias_adjustment = self.adjust(loss, credit_score)  # Step 3) Adjust(Calc)
        new_weight = self.weight + weight_adjustment                   # Step 3) Adjust(Apply)
        new_bias = self.bias + bias_adjustment                         # Step 3) Adjust(Apply)
        gladiator_output = GladiatorOutput(
            prediction=prediction,
            adjustment=weight_adjustment,
            weight=self.weight,
            new_weight=new_weight,
            bias=self.bias,
            new_bias=new_bias
        )
        self.weight = new_weight
        self.bias = new_bias
        return gladiator_output

    def predict(self, credit_score: float) -> int:
        # Include bias in the prediction
        return 1 if round(credit_score * self.weight + self.bias, 7) >= 0.5 else 0  # Rounded to 7 decimals to avoid FP errors

    def compare(self, prediction: float, result: float) -> float:      # Calculate the Loss
        return result - prediction

    def adjust(self, loss: float, input_value: float) -> tuple:        # Apply learning rate to loss for both weight and bias
        weight_adjustment = loss * self.learning_rate * input_value
        bias_adjustment = loss * self.learning_rate
        return weight_adjustment, bias_adjustment