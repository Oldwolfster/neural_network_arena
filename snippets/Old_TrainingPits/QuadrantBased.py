from typing import List, Tuple
from src.TrainingPit import TrainingPit
import random


class QuadrantBased(TrainingPit):
    """
    Concrete class that generates linearly separable training data based on quadrants.
    Points in the upper-right or lower-left quadrants are considered positive cases (1).
    If include_anomalies is true, points are misclassified based on their distance from the origin.
    """

    def __init__(self, num_samples: int, include_anomalies: bool):
        self.num_samples = num_samples
        self.include_anomalies = include_anomalies

    def generate_training_data(self) -> List[Tuple[int, int, int]]:
        training_data = []
        for _ in range(self.num_samples):
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if self.include_anomalies:
                distance = (x ** 2 + y ** 2) ** 0.5
                probability = distance / 141.42  # Normalize distance (max distance from origin is sqrt(2)*100)
                label = 1 if (random.random() > probability and (x * y > 0)) else 0
            else:
                label = 1 if (x > 0 and y > 0) or (x < 0 and y < 0) else 0
            training_data.append((x, y, label))
        return training_data

