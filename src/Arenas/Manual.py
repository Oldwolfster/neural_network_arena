from src.BaseArena import TrainingPit
import random
from typing import List, Tuple
class Manual(TrainingPit):
    """
    This class allows you to send literal training data
    for example if you need repeatable results
    """
    def __init__(self,num_samples: int):
        self.num_samples = num_samples
    def generate_training_data(self) -> List[Tuple[float, float]]:
        return [(13, 0), (27, 0), (13, 1), (73, 1), (82, 0)]