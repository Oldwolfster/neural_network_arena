from src.TrainingPit import TrainingPit
import random
from typing import List, Tuple


class SingleInput_CreditScoreShifted(TrainingPit):
    """
    Generates training data with an offset decision boundary.
    This data should perform significantly better with a model that includes bias.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples
        self.offset = 8  # Reduced offset value
        self.scale = 10  # Scale factor for inputs

    def generate_training_data(self) -> List[Tuple[float, int]]:
        training_data = []
        for _ in range(self.num_samples):
            # Generate score between 0 and 10
            score = random.uniform(0, 10)
            repayment_chance = score / 10  # Normalize to 0-1
            second_number = 1 if random.random() < repayment_chance else 0

            # Scale the score
            scaled_score = score * self.scale
            training_data.append((scaled_score, second_number))
        return training_data