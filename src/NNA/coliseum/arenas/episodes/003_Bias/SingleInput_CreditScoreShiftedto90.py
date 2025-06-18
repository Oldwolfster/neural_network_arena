from src.NNA.engine.BaseArena import BaseArena
import random
from typing import List, Tuple


class SingleInput_CreditScoreShiftedto90(BaseArena):
    """
    Concrete class that generates training data with a more extremely shifted decision boundary.
    This data will perform better with a model that includes bias.

    The credit score is between 0-100, but the decision boundary is shifted to 90.
    If include_anomalies is false, scores of 90 or greater result in repayment (1).
    If include_anomalies is true, it uses (score - 40) as the percent chance the loan was repaid.
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[int, int]]:
        training_data = []
        for _ in range(self.num_samples):
            score = random.randint(1, 100)
            second_number = 1 if random.random() < .1 else 0
            training_data.append((score, second_number))
        return training_data