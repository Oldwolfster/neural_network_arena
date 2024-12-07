from src.engine.Metrics import GladiatorOutput
from src.gladiators.BaseGladiator import Gladiator
import math

class _Template_Simpletron_Claude2(Gladiator):
    """
    An enhanced perceptron implementation for educational purposes.
    This class serves as a template for more complex implementations.

    The training data is a tuple where the first element is the input and the second is the target
    After processing return a GladiatorOutput object with the fields populated. Neural Network Arena will handle the rest
    """

    def __init__(self, number_of_epochs: int, metrics_mgr: MetricsMgr, *args):
        super().__init__(number_of_epochs, metrics_mgr, *args)
        self.bias = 0.0  # Initialize bias
        self.epsilon = 1e-15  # Small constant to avoid division by zero

    def training_iteration(self, training_data) -> GladiatorOutput:
        credit_score = training_data[0]
        result = training_data[1]
        raw_prediction = self.raw_predict(credit_score)               # Step 1) Raw prediction
        prediction = self.activate(raw_prediction)                    # Step 1) Apply activation function
        loss = self.calculate_loss(prediction, result)                # Step 2) Calculate loss
        weight_adjustment, bias_adjustment = self.adjust(loss, credit_score, raw_prediction)  # Step 3) Adjust(Calc)
        new_weight = self.weight + weight_adjustment                  # Step 3) Adjust(Apply)
        new_bias = self.bias + bias_adjustment                        # Step 3) Adjust(Apply)
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

    def raw_predict(self, credit_score: float) -> float:
        return credit_score * self.weight + self.bias

    def activate(self, x: float) -> float:
        return 1 / (1 + math.exp(-x))  # Sigmoid activation function

    def predict(self, credit_score: float) -> int:
        raw_prediction = self.raw_predict(credit_score)
        return 1 if self.activate(raw_prediction) >= 0.5 else 0

    def calculate_loss(self, prediction: float, result: float) -> float:
        # Binary cross-entropy loss
        return -1 * (result * math.log(prediction + self.epsilon) + (1 - result) * math.log(1 - prediction + self.epsilon))

    def adjust(self, loss: float, input_value: float, raw_prediction: float) -> tuple:
        # Gradient of sigmoid
        sigmoid_gradient = self.activate(raw_prediction) * (1 - self.activate(raw_prediction))

        # Gradient of loss with respect to raw prediction
        loss_gradient = self.activate(raw_prediction) - input_value

        # Adjustments using chain rule
        weight_adjustment = -self.learning_rate * loss_gradient * sigmoid_gradient * input_value
        bias_adjustment = -self.learning_rate * loss_gradient * sigmoid_gradient

        return weight_adjustment, bias_adjustment