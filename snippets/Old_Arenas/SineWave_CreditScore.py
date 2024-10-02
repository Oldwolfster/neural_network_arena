import random
import math
from typing import List, Tuple
from src.BaseArena import TrainingPit

class SineWave_CreditScore(TrainingPit):
    """
    Generates non-linear training data based on a sine wave function.
    Introduces periodicity into the relationship between credit score and repayment.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[int, int]]:
        training_data = []
        for _ in range(self.num_samples):
            score = random.randint(1, 100)
            probability = (math.sin(score / 10) + 1) / 2  # Normalize sine wave to [0, 1]
            second_number = 1 if random.random() < probability else 0
            training_data.append((score, second_number))
        return training_data
