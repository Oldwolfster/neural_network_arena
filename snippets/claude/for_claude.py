I’ve been on a quest to build the simplest neural network possible for the purpose of education and I’ve named it Simpletron.  It has no bias and doesn’t  do gradient descent but gets the job done.
I’ve also created the “Arena”, a system consisting of 3 main parts
Gladiators (Neural Network Models)
Training Pits (algorithms to produce training/test data)
Arena engine which
1) gets a single set of data from the Training pit
2) sends it to all competing Gladiators
3)Compiles and reports on the results.
Here’s my current plan of how I am proceeding,

Phase 1: Single Neuron, Single Input

1. Introduction to Simpletron (done)
2. Learning Rate and Convergence (done)
3. Bias in Neural Networks (done)
4. From Binary to Regression(next in line)
5. Activation Functions (Linear vs Non-linear)
6. Cost Functions and their Impact  Cross Entropy, MSE, MAE,  BCE, CCE, Hinge, Huber, KL Divergence
6.5. Gradient Descent Variations (e.g., Stochastic vs Batch)

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

from src.Arena import *
from src.Metrics import Metrics, IterationData
from src.Gladiator import Gladiator

class _Template_Simpletron_Regressive(Gladiator):
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
        for i, (sqr_ft, price) in enumerate(train_data):                # Loop through all the training data
            self.training_iteration(i, epoch_num, sqr_ft, price)        # Run single sample of training data
        return self.metrics.record_epoch()                              # Sends back the data for an epoch

    def training_iteration(self, i: int, epoch: int, sqr_ft: float, price: float) -> None:
        prediction  = self.predict(sqr_ft)                              # Step 1) Guess
        error       = self.compare(prediction, price)                   # Step 2) Check guess, if wrong, by how much and quare it (MSE)
        adjustment  = self.adjust_weight(error)                         # Step 3) Adjust(Calc)
        loss        = (error ** 2) * 0.5                                # For MSE the loss is the error squared and cut in half
        new_weight  = self.weight + adjustment                          # Step 3) Adjust(Apply)
        data = IterationData(
            iteration=i+1,
            epoch=epoch+1,
            input=sqr_ft,
            target=price,
            prediction=prediction,
            adjustment=adjustment,
            weight=self.weight,
            new_weight=new_weight,
            bias=0.0,
            old_bias=0.0
        )

        self.metrics.record_iteration(data)


        #self.metrics.record_iteration(i, epoch, credit_score, result, prediction, loss, adjustment, self.weight, new_weight, self.metrics)
        # def record_iteration(self, iteration, epoch, input_value, result, prediction, loss, adjustment, weight, new_weight, metrics, bias=0, old_bias=0):
        self.weight = new_weight

    def predict(self, credit_score: float) -> float:
        return credit_score * self.weight                               # Simplifies prediction as no step function


    def compare(self, prediction: float, target: float) -> float:            # Calculate the Loss (Just MSE for now)
        return (target - prediction)

    def adjust_weight(self, loss: float) -> float:                       # Apply learning rate to loss.
        # Chat GPT says it should be 2 * Loss * Learning Rate * input
        return loss * self.learning_rate

Here is an example of a Training Pit
from src.TrainingPit import TrainingPit
import random
from typing import List, Tuple
class HouseValue_SqrFt(TrainingPit):
    """
    Concrete class that generates  training data for regression.
    it first calculates square feet between 500-4000.
    It then determines price per sq ft, using 200 as a constant
    It adds noise from a gaussian distribution centered on 0 with 68% of values within 50k
    finally, it double checks the value isn't negative
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            # Generate square footage between 500 and 4000
            sqft = round(random.uniform(1500, 4000), 0)

            # Base price per square foot
            price_per_sqft = 200  # Adjust as needed

            # Calculate base price
            base_price = sqft * price_per_sqft

            # Try adding a flat amount like 20k and see if that makes the bias useful?

            # Add noise to the price
            noise = random.gauss(0, 50000)  # Mean 0, SD 50,000
            price = base_price + noise
            #price = base_price # temporarily remove the noise
            # Ensure price doesn't go below zero and round to nearest dollar
            #price = round(max(price, random.uniform(50000, 40000)),0)
            #price += 5000000 # I was expecting this to make use of bias
            training_data.append((sqft, price))
        return training_data