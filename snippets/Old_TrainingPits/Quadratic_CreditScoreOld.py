import random
from typing import List, Tuple
from src.TrainingPit import TrainingPit

class Quadratic_CreditScore(TrainingPit):
    """
    Generates non-linear training data where the relationship follows a quadratic curve.
    Medium credit scores have a higher likelihood of repayment, forming a bell-shaped curve.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[int, int]]:
        training_data = []
        for _ in range(self.num_samples):
            score = random.randint(1, 100)
            probability = -0.02 * (score - 50) ** 2 + 1  # Quadratic relationship
            second_number = 1 if random.random() < probability else 0
            training_data.append((score, second_number))
        return training_data
