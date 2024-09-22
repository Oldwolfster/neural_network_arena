from src.Arena import *
from src.Metrics import Metrics, IterationData
from src.Gladiator import Gladiator

class Regressive_Bias(Gladiator):
    """
    A simple perceptron implementation for educational purposes.
    This class is setup for regression using both a weight and bias

    Leverage metrics with just two steps required:
    1) After each iteration call metrics.record_iteration_metrics
    2) After each iteration call metrics.record_epoch_metrics (no additional info req - uses info from iterations)
    """

    def __init__(self, number_of_epochs: int, metrics: Metrics, *args):
        super().__init__(number_of_epochs, metrics, *args)
        self.bias = 1

    def train(self, training_data):
        for epoch in range(self.number_of_epochs):                      # Loop to run specified # of epochs
            if self.run_an_epoch(training_data, epoch):                 # Call function to run single epoch
                break

    def run_an_epoch(self, train_data, epoch_num: int) -> bool:         # Function to run single epoch
        for i, (sqr_ft, price) in enumerate(train_data):                # Loop through all the training data
            self.training_iteration(i, epoch_num, sqr_ft, price)        # Run single sample of training data
        return self.metrics.record_epoch()                              # Sends back the data for an epoch

    def training_iteration(self, i: int, epoch: int, sqr_ft: float, price: float) -> None:
        prediction  = self.predict(sqr_ft)                              # Step 1) Guess
        error       = self.compare(prediction, price)                   # Step 2) Check guess, if wrong, by how much and quare it (MSE)
        adjustment  = self.adjust_weight(error)                         # Step 3) Adjust(Calc)
        loss        = (error ** 2) * 0.5                                # For MSE the loss is the error squared and cut in half
        new_weight  = self.weight + adjustment                          # Step 3) Adjust(Apply)
        new_bias    = self.bias + adjustment
        data = IterationData(
            iteration=i+1,
            epoch=epoch+1,
            input=sqr_ft,
            target=price,
            prediction=prediction,
            adjustment=adjustment,
            weight=self.weight,
            new_weight=new_weight,
            bias=new_bias,
            old_bias=self.bias
        )

        #print(f"new_bias{new_bias}\t old bias{self.bias}")


        self.metrics.record_iteration(data)


        #self.metrics.record_iteration(i, epoch, credit_score, result, prediction, loss, adjustment, self.weight, new_weight, self.metrics)
        # def record_iteration(self, iteration, epoch, input_value, result, prediction, loss, adjustment, weight, new_weight, metrics, bias=0, old_bias=0):
        self.weight = new_weight
        self.bias   = new_bias

    def predict(self, credit_score: float) -> float:
        return credit_score * self.weight + self.bias                             # Simplifies prediction as no step function


    def compare(self, prediction: float, target: float) -> float:            # Calculate the Loss (Just MSE for now)
        return (target - prediction)

    def adjust_weight(self, loss: float) -> float:                       # Apply learning rate to loss.
        # Chat GPT says it should be 2 * Loss * Learning Rate * input
        return loss * self.learning_rate
