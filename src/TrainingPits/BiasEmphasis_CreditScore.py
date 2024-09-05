from src.TrainingPit import TrainingPit
import random
from typing import List, Tuple


class BiasEmphasis_CreditScore(TrainingPit):
    """
    Concrete class that generates training data to emphasize the impact of bias.
    It calculates a credit score between 600-800, shifting the data away from the origin.
    The decision boundary is set at 700.
    If include_anomalies is false, scores >= 700 result in repayment (1), otherwise no repayment (0).
    If include_anomalies is true, it uses (score - 600) / 200 as the probability of repayment.
    """

    def __init__(self, num_samples: int, include_anomalies: bool):
        self.num_samples = num_samples
        self.include_anomalies = include_anomalies

    def generate_training_data(self) -> List[Tuple[float, int]]:
        training_data = []
        for _ in range(self.num_samples):
            score = random.uniform(600, 800)
            if self.include_anomalies:
                repayment_probability = (score - 600) / 200
                result = 1 if random.random() < repayment_probability else 0
            else:
                result = 1 if score >= 700 else 0

            # Normalize the score to be between 0 and 1
            normalized_score = (score - 600) / 200

            training_data.append((normalized_score, result))
        return training_data