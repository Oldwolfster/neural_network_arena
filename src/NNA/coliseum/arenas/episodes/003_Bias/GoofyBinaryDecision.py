from src.NNA.engine.BaseArena import BaseArena
import random
from typing import List, Tuple


class BiasEmphasis_CreditScore(BaseArena):
    """
    Concrete class that generates training data for binary decision.
    It calculates a credit score between 0-100, .
    That is the percentage chance that it will be repaid
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, int]]:
        training_data = []
        for _ in range(self.num_samples):
            score = random.uniform(0, 100)

            repayment_probability = score  / 100
            result = 8 if random.random() < repayment_probability else 2

            training_data.append((score, result))
        return training_data, ["Credit Score", "Repayment"]
