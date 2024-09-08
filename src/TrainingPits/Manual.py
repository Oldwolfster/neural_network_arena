from src.TrainingPit import TrainingPit
import random
from typing import List, Tuple
class Manual(TrainingPit):
    """
    Concrete class that generates linearly separable training data.
    it first calculates a credit score between 0-100.  If include_anomalies is false and the credit is 50 or greater the output is 1 (repayment)
    if include_anomalies is true, it uses the credit score as the percent chance the loan was repaid
    for example a score of 90 would normally repay, but there is a 10% chance it will not.
    """
    def __init__(self,num_samples: int):
        self.num_samples = num_samples
    def generate_training_data(self) -> List[Tuple[int, int]]:
        return [(84, 1), (38, 1), (55, 1), (84, 1), (17, 0), (84, 1), (73, 1), (38, 0), (5, 0), (42, 1), (26, 0), (1, 0), (7, 0), (75, 0), (10, 0), (46, 0), (28, 0), (51, 0), (34, 0), (8, 0), (21, 0), (5, 0), (70, 1), (3, 0), (35, 0), (72, 1), (82, 1), (36, 0), (92, 0), (97, 1)]