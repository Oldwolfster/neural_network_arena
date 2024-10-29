import random
from typing import List, Tuple
from src.engine.BaseArena import BaseArena


class Salary2InputsLinear(BaseArena):
    """
    Generates regression training data with a linear relationship between inputs and salary.
    This makes it solvable by a single neuron perceptron without adding extra inputs.
    NOTE: optimum bias should be the base salary!

    To make the salary calculation solvable by a single neuron perceptron with two inputs, we need to ensure that the relationship between the inputs and the salary is linear. The original calculation includes a multiplicative interaction term (salary = ... * (college + 0.5)), introducing non-linearity.
    Loss Function: Mean squared error (since it's a regression task).
    Salary=Base Salary+(Coefficient1  ×Years of Experience)+(Coefficient2 × College)+Noise
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def generate_training_data(self) -> List[Tuple[float, float, float]]:
        training_data = []
        for _ in range(self.num_samples):
            years_experience = random.uniform(0, 40)
            college = random.uniform(0, 8)
            # Linear salary calculation
            base_salary = 30  # in thousands
            coeff_experience = 4  # in thousands per year
            coeff_college = 5  # in thousands per year of college

            #noise = random.gauss(0, 5)  # Add some noise for variance
            noise = 0  # Add some noise for variance AFTER IT WORKS WITHOUT

            salary = base_salary + (coeff_experience * years_experience) + (coeff_college * college) + noise

            training_data.append((years_experience, college, salary))
        return training_data
