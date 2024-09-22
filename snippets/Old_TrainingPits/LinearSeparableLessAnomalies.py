from abc import ABC, abstractmethod
from typing import List, Tuple, Any
from src.TrainingPit import TrainingPit
import random
from typing import List, Tuple

class LinearSeparableLessAnomalies(TrainingPit):
    """
    Concrete class that generates linearly separable training data with a single input and single output.
    The input is a single number between 0 and 100, and the output is 1 if the input is greater than or equal to 50, and 0 otherwise.
    If 'include_anomalies' is True, the output will be flipped with a 10% probability.
    """
    def __init__(self, num_samples: int, include_anomalies: bool):
        self.num_samples = num_samples
        self.include_anomalies = include_anomalies

    def generate_training_data(self) -> List[Tuple[int, int]]:
        training_data = []
        for _ in range(self.num_samples):
            # Generate input feature
            input_feature = random.randint(0, 100)

            # Calculate the target output
            target = 1 if input_feature >= 50 else 0

            # Introduce anomalies if requested
            if self.include_anomalies:
                if random.random() < 0.1:  # 10% chance of anomaly
                    target = 1 - target

            training_data.append((input_feature, target))

        return training_data