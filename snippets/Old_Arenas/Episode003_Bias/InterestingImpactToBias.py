from src.BaseArena import TrainingPit
import random
from typing import List, Tuple


class InterestingImpactToBias(TrainingPit):
    """
    Concrete class that generates training data with a decision boundary at 90.
    This data will perform better with a model that includes bias.

    The credit score is between 0-100, and the decision boundary is shifted to 90.
    Scores of 90 or greater result in repayment (1). If include_anomalies is true,
    scores are randomly adjusted around the boundary to introduce noise.
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples
        self.boundary = 90  # Decision boundary at 90

    def generate_training_data(self) -> List[Tuple[int, int]]:
        training_data = []
        for _ in range(self.num_samples):
            score = random.randint(1, 100)

            # Introduce some randomness around the boundary to create noise
            # We use a small random adjustment around the boundary
            adjustment = random.uniform(-5, 5)  # Adjust by up to 5 points
            adjusted_score = score + adjustment
            second_number = 1 if adjusted_score >= self.boundary else 0

            training_data.append((score, second_number))
        return training_data
