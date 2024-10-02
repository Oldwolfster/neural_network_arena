import random
from typing import List, Tuple
from src.BaseArena import TrainingPit

class StepFunction_CreditScore(TrainingPit):
    """
    Generates non-linear training data based on a step function.
    Different ranges of credit scores have distinct probabilities of repayment.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[int, int]]:
        training_data = []
        for _ in range(self.num_samples):
            score = random.randint(1, 100)
            if score < 30:
                probability = 0.2
            elif 30 <= score < 60:
                probability = 0.5
            else:
                probability = 0.8
            second_number = 1 if random.random() < probability else 0
            training_data.append((score, second_number))
        return training_data
