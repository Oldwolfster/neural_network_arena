from src.TrainingPit import TrainingPit
import random
from typing import List, Tuple

class QuadraticTrainingPit(TrainingPit):
    """
    Generates training data for a quadratic function y = x^2.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            x = random.uniform(-1, 1)  # Input range [-1, 1]
            y = x ** 2  # Quadratic relationship
            training_data.append((x, y))
        return training_data
