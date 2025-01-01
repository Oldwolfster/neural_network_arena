from src.ArenaSettings import HyperParameters
from src.engine.TrainingData import TrainingData


class ConvergenceDetector:
    def __init__(self, hyper: HyperParameters, td: TrainingData):
        self.hyper = hyper
        self.td = td
        self.relative_threshold = self.calculate_relative_threshold()
        self.MAE_per_epoch = []

    def calculate_relative_threshold(self) -> float:
        mean_of_targets = self.td.sum_of_targets / self.td.sample_count
        return mean_of_targets * self.hyper.converge_threshold / 100

    def check_convergence(self, MAE: float) -> bool:
        self.MAE_per_epoch.append(MAE)
        if len(self.MAE_per_epoch) < 2:
            return False  # Not enough data to evaluate
        change = abs(self.MAE_per_epoch[-1] - self.MAE_per_epoch[-2])
        print (f"change")

        return change < self.relative_threshold
