import random
from typing import List, Tuple
from src.engine.BaseArena import BaseArena

class XOR_Floats(BaseArena):
    """
    The classic XOR problem with float noise to break symmetry.
    Impossible for models without a hidden layer or nonlinearity.
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            x = random.choice([0, 1]) + random.uniform(-0.2, 0.2)
            y = random.choice([0, 1]) + random.uniform(-0.2, 0.2)
            label = 1.0 if (round(x) != round(y)) else 0.0
            training_data.append((x, y, label))
        return training_data, ["Input A", "Input B", "XOR Label"]