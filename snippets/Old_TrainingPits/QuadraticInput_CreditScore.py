from src.TrainingPit import TrainingPit
import random
from typing import List, Tuple

class QuadraticInput_CreditScore(TrainingPit):
    """
    Generates quadratic data where the output is based on a non-linear relationship with the input.
    A score of 50 or higher will output a 1, but the probability decreases quadratically as the score decreases.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[int, int]]:
        training_data = []
        for _ in range(self.num_samples):
            score = random.randint(1, 100)
            second_number = 1 if random.random() < ((score / 100) ** 2) else 0  # Quadratic relation
            training_data.append((score, second_number))
        return training_data
