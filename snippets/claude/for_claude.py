Hello, I’ve been on a quest to build the simplest neural network possible for the purpose of education and I’ve named it Simpletron.  It has no bias and doesn’t  do gradient descent but gets the job done.
I’ve also created the “Arena”, a system consisting of 3 main parts
Gladiators (Neural Network Models)
Training Pits (algorithms to produce training/test data)
Arena engine – 1) gets one set of data from the Training pit 2) sends it to all competing Gladiators  3)Compiles and reports on the results.
Here’s my current plan of how I am proceeding,

Phase 1: Single Neuron, Single Input

1. Introduction to Simpletron (already covered)
2. Learning Rate and Convergence (upcoming video)
3. Bias in Neural Networks
4. Activation Functions (Linear vs Non-linear)
5. Binary vs Regression(Continuous values)
5. Cost Functions and their Impact  Cross Entropy, MSE, MAE,  BCE, CCE, Hinge, Huber, KL Divergence
6. Gradient Descent Variations (e.g., Stochastic vs Batch)

Phase 2: Multiple Inputs

7. From Single to Multiple Inputs
8. Feature Scaling and Normalization
9. The XOR Problem and its Significance (Single layer perceptron cannot solve)
10. Multivariate Regression vs Classification

Phase 3: Multiple Neurons (Single Layer)

11. Introduction to Multi-Layer Perceptrons
12. Backpropagation Explained
13. Vanishing and Exploding Gradients
14. Regularization Techniques
15. Dropout and Other Modern Techniques
Phase 4: Multiple Layers

Bonus Videos (can be interspersed)

The Math Behind Neural Networks
Implementing NNs from Scratch in Different Languages
Real-world Applications of Simple NNs
Common Mistakes and How to Avoid Them
Neural Network "Battle Royale": Comparing Different Architectures

Here’s an example of a Gladiator
from src.Arena import *
from src.Metrics import Metrics
from src.Gladiator import Gladiator

class _Template_Simpletron(Gladiator):
    """
    A simple perceptron implementation for educational purposes.
    This class serves as a template for more complex implementations.

    In order to utilize the metrics there are two steps required:
    1) After each iteration call metrics.record_iteration_metrics
    2) After each iteration call metrics.record_epoch_metrics (no additional info req - uses info from iterations)
    """

    def __init__(self, number_of_epochs: int, metrics: Metrics, *args):
        super().__init__(number_of_epochs, metrics, *args)
       #  print(f'Learning rate2 = {self.learning_rate} args[1]{args[1]}')
        # Ideally avoid overriding these, but specific models may need, so must be free to do so
        # It keeps comparisons straight if respected
        # self.weight = override_weight
        # self.learning_rate = override_learning_rate

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

Here is an example of a Training Pit
from src.TrainingPit import TrainingPit
import random
from typing import List, Tuple
class SingleInput_CreditScore(TrainingPit):
    """
    Concrete class that generates linearly separable training data.
    it first calculates a credit score between 0-100.  If include_anomalies is false and the credit is 50 or greater the output is 1 (repayment)
    if include_anomalies is true, it uses the credit score as the percent chance the loan was repaid
    for example a score of 90 would normally repay, but there is a 10% chance it will not.
    """
    def __init__(self,num_samples: int, include_anomalies: bool):
        self.num_samples = num_samples
        self.include_anomalies = include_anomalies
    def generate_training_data(self) -> List[Tuple[int, int]]:
        training_data = []
        for _ in range(self.num_samples):
            score = random.randint(1, 100)
            if self.include_anomalies:
                second_number = 1 if random.random() < (score / 100) else 0
            else:
                second_number = 1 if score >= 50 else 0
            training_data.append((score, second_number))
        return training_data
