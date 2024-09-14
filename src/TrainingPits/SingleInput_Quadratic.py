from src.TrainingPit import TrainingPit
import random
import math
from typing import List, Tuple


class SingleInput_Quadratic(TrainingPit):
    """
    Generates training data with a complex, non-linear relationship between credit score and repayment probability.
    This relationship is designed to benefit from ReLU activation in a single neuron model.
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[int, int]]:
        training_data = []
        for _ in range(self.num_samples):
            score = random.randint(1, 100)

            # Calculate repayment probability using a complex function
            repayment_prob = self._complex_probability(score)

            # Generate outcome based on probability
            repayment = 1 if random.random() < repayment_prob else 0

            training_data.append((score, repayment))

        return training_data

    def _complex_probability(self, score: int) -> float:
        # Complex function to calculate repayment probability
        if score < 20:
            return 0
        elif 20 <= score < 50:
            return 0.1 * math.sin(score / 5)
        elif 50 <= score < 80:
            return 0.5 + 0.3 * math.log(score - 49)
        else:
            return min(1, 0.8 + 0.002 * (score - 80) ** 2)