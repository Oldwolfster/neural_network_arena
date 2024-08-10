from abc import ABC, abstractmethod

from src.Metrics import Metrics


class Gladiator(ABC):
    def __init__(self, number_of_epochs: int, metrics: Metrics, *args):
        self.number_of_epochs = number_of_epochs
        self.metrics = metrics
        self.weight = args[0]
        self.learning_rate = args[1]

    @abstractmethod
    def train(self, training_data):
        pass
