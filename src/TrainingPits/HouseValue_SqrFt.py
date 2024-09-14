from src.TrainingPit import TrainingPit
import random
from typing import List, Tuple
class HouseValue_SqrFt(TrainingPit):
    """
    Concrete class that generates  training data for regression.
    it first calculates square feet between 500-4000.
    It then determines price per sq ft, using 200 as a constant
    It adds noise from a gaussian distribution centered on 0 with 68% of values within 50k
    finally, it double checks the value isn't negative
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            # Generate square footage between 500 and 4000
            sqft = round(random.uniform(500, 4000), 0)

            # Base price per square foot
            price_per_sqft = 200  # Adjust as needed

            # Calculate base price
            base_price = sqft * price_per_sqft

            # Add noise to the price
            noise = random.gauss(0, 50000)  # Mean 0, SD 50,000
            price = base_price + noise

            # Ensure price doesn't go below zero and round to nearest dollar
            price = round(max(price, 0),0)

            training_data.append((sqft, price))
        return training_data