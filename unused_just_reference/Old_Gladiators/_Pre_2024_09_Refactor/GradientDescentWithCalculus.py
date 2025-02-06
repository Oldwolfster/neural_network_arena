from src.engine.Metrics import Metrics
from src.engine.BaseGladiator import Gladiator

class GradientDescentWithCalculus(Gladiator):
    """
    A perceptron implementation that explicitly uses calculus for gradient descent.
    This version uses Mean Squared Error (MSE) as the loss function and updates weights and bias accordingly.
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
        loss = self.compute_loss(prediction, result)                    # Step 2) Calculate the MSE Loss
        gradient_w, gradient_b = self.compute_gradients(loss, credit_score)  # Step 3) Compute Gradients
        new_weight = self.weight - self.learning_rate * gradient_w      # Step 4) Update Weight using gradient
        new_bias = self.bias - self.learning_rate * gradient_b          # Step 4) Update Bias using gradient
        self.metrics.record_iteration(i, epoch, credit_score, result, prediction, loss, gradient_w, self.weight, new_weight, self.metrics)
        self.weight = new_weight
        self.bias = new_bias

    def predict(self, credit_score: float) -> float:
        return credit_score * self.weight + self.bias

    def compute_loss(self, prediction: float, result: float) -> float:
        # Mean Squared Error (MSE) Loss
        return (prediction - result) ** 2

    def compute_gradients(self, loss: float, credit_score: float) -> tuple:
        # Derivative of MSE with respect to weight and bias
        gradient_w = 2 * (self.predict(credit_score) - credit_score) * credit_score
        gradient_b = 2 * (self.predict(credit_score) - credit_score)
        return gradient_w, gradient_b
