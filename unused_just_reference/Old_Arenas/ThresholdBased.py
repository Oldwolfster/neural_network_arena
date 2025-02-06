import random
from typing import List, Tuple
from src.BaseArena import TrainingPit


class ThresholdBased(TrainingPit):
    """
    Concrete class that generates linearly separable training data based on a threshold.
    If include_anomalies is true, the score is used as the percent chance the output is correct.
    """

    def __init__(self, num_samples: int, include_anomalies: bool):
        self.threshold = 75
        self.num_samples = num_samples
        self.include_anomalies = include_anomalies
    def generate_training_data(self) -> List[Tuple[int, int]]:
        training_data = []
        for _ in range(self.num_samples):
            score = random.randint(1, 100)
            if self.include_anomalies:
                label = 1 if random.random() < (score / 100) else 0
            else:
                label = 1 if score >= self.threshold else 0
            training_data.append((score, label))
        return training_data



