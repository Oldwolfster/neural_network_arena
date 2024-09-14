from src.TrainingPit import TrainingPit
import random
from typing import List, Tuple
class SingleInput_CreditScore(TrainingPit):
    """
    Concrete class that generates linearly separable training data.
    it first calculates a credit score between 0-100. It then uses the credit score as the percent chance the loan was repaid
    for example a score of 90 would normally repay, but there is a 10% chance it will not.
    """
    def __init__(self,num_samples: int):
        self.num_samples = num_samples
    def generate_training_data(self) -> List[Tuple[int, int]]:
        training_data = []
        for _ in range(self.num_samples):
            score = random.randint(1, 100)
            second_number = 1 if random.random() < (score / 100) else 0
            training_data.append((score, second_number))
        return training_data
