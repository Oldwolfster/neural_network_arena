import random
import math
from typing import List, Tuple
from src.engine.BaseArena import BaseArena


class Adversarial_Noise(BaseArena):
    """
    Regression task with adversarial twist:
    Gaussian noise is injected into the target halfway through the dataset,
    simulating mid-training data corruption.

    Useful for testing optimizer stability and generalization.
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> Tuple[List[Tuple[float, float]], List[str]]:
        training_data = []

        for i in range(self.num_samples):
            x = random.uniform(-10, 10)
            y = 3 * x + 7  # Clean linear relation

            if i > self.num_samples // 2:
                y += random.gauss(0, 15)  # Inject adversarial noise

            training_data.append((x, y))

        return training_data, ["Input X", "Target Y"]
