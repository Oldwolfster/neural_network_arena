from src.engine.Metrics import GladiatorOutput
from src.engine.BaseGladiator import Gladiator

class Regression_GBS_2_Inputs(Gladiator):
    """
    A simple perceptron implementation for accurate regression. (By ChatGPT)
    It is designed for training data that predicts repayment ratio (0.0 to 1.0)
    based on credit score between 0-100, with added noise.
    Includes bias and improved weight adjustment logic for accuracy.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.bias = .5
        self.wgt2 = self.weight  #start at same weight

    def training_iteration(self, training_data) -> GladiatorOutput:
        input = training_data[0]
        inp2  = training_data[1]
        target = training_data[-1]
        #prediction = (input * self.weight) + self.bias
        prediction = (input * self.weight) + (inp2 * self.wgt2) + self.bias
        error = target - prediction
        new_weight = self.weight + self.learning_rate * error * input
        new_wgt2 = self.wgt2     + self.learning_rate * error * input
        new_bias = self.bias     + self.learning_rate * error  # Bias adjustment

        # Output object containing results of this iteration
        gladiator_output = GladiatorOutput(
            prediction=prediction,
            adjustment=new_weight-self.weight,
            weight=self.weight,
            new_weight=new_weight,
            bias=self.bias,
            new_bias=new_bias
        )

        # Update model parameters
        self.weight = new_weight
        self.wgt2 = new_wgt2
        self.bias = new_bias

        return gladiator_output

    def predict(self, credit_score: float) -> float:
        """
        Predict repayment ratio based on the credit score and current weights.
        Includes bias term.
        """
        return (credit_score * self.weight) + self.bias  # Adding bias term to the prediction

    def compare(self, prediction: float, target: float) -> float:
        """
        Calculate the error: difference between target and prediction.
        """
        return target - prediction

    def adjust_weight(self, error: float, credit_score: float) -> float:
        """
        Adjust weight based on the error, learning rate, and input.
        Following gradient descent principles: adjustment = error * learning_rate * input
        """
        return self.learning_rate * error * credit_score
