from abc import ABC, abstractmethod
from typing import List, Tuple, Any
from src.TrainingPit import TrainingPit
import random
from typing import List, Tuple


class LinearSeparable(TrainingPit):
    """
    Concrete class that generates linearly separable training data.
    it first calculates a credit score between 0-100.  If include_anomolies is false and the credit is 50 or greater the output is 1 (repayment)
    if include_anomalies is true, it uses the credit score as the percent chance the loan was repaid
    for example a score of 90 would normally repay, but there is a 10% chance it will not.
    """
    def __init__(self,num_samples: int, include_anomalies: bool):
        self.num_samples = num_samples
        self.include_anomalies = include_anomalies
    def generate_training_data(self) -> List[Tuple[int, int]]:
        training_data = []
        for _ in range(self.num_samples):
            score = random.randint(1, 100)
            if self.include_anomalies:
                second_number = 1 if random.random() < (score / 100) else 0
            else:
                second_number = 1 if score >= 50 else 0
            training_data.append((score, second_number))
        return training_data
