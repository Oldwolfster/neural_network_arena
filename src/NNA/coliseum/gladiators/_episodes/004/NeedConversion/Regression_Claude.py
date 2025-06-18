from src.NNA.engine.Metrics import GladiatorOutput
from src.NNA.engine.BaseGladiator import Gladiator
import numpy as np

class Regression_Claude(Gladiator):
    """
    An improved perceptron implementation for educational purposes.
    This class enhances the template with regularization, bias, and adaptive learning rate.
    It is designed for general regression tasks and can handle any range of target values.
    """

    def __init__(self, number_of_epochs: int, metrics_mgr: MetricsMgr, *args):
        super().__init__(number_of_epochs, metrics_mgr, *args)
        self.bias = 0.0
        self.lambda_reg = 0.01  # L2 regularization strength
        self.adaptive_lr = 0.01  # Initial adaptive learning rate
        self.epsilon = 1e-8  # Small value to avoid division by zero

    def training_iteration(self, training_data) -> GladiatorOutput:
        input_feature = training_data[0]
        target = training_data[1]
        prediction = self.predict(input_feature)
        error = self.compare(prediction, target)
        weight_adjustment, bias_adjustment = self.adjust(error, input_feature)
        new_weight = self.weight + weight_adjustment
        new_bias = self.bias + bias_adjustment

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

    def predict(self, input_feature: float) -> float:
        return input_feature * self.weight + self.bias

    def compare(self, prediction: float, target: float) -> float:
        return target - prediction

    def adjust(self, error: float, input_feature: float) -> tuple:
        # Gradient for weight
        dw = error * input_feature

        # Gradient for bias
        db = error

        # L2 regularization term
        l2_reg = self.lambda_reg * self.weight

        # Adaptive learning rate using RMSprop
        self.adaptive_lr = 0.9 * self.adaptive_lr + 0.1 * dw**2

        # Calculate adjustments
        weight_adjustment = (self.learning_rate / (np.sqrt(self.adaptive_lr) + self.epsilon)) * (dw - l2_reg)
        bias_adjustment = self.learning_rate * db

        return weight_adjustment, bias_adjustment