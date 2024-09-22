from src.TrainingPit import TrainingPit
import random
from typing import List, Tuple
import math


class HouseValue_SqrFt__ForBias(TrainingPit):
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            sqft = round(random.uniform(500, 4000), 0)
            price_per_sqft = random.normalvariate(200, 20)

            # Introduce non-linearity
            base_price = (sqft * price_per_sqft) + (math.sqrt(sqft) * 1000)

            # Add a random large offset
            #base_price += random.uniform(1e9, 5e9) #PAP of 40
            #base_price += random.uniform(1e6, 5e6)  # PAP of 53
            base_price += random.uniform(1e3, 5e3)  # PAP of 53
            noise = random.gauss(0, 50000)
            price = base_price + noise

            if price < 10000:
                price = 10000 + random.uniform(0, 40000)

            price = round(price, 0)
            training_data.append((sqft, price))
        return training_data