from abc import ABC, abstractmethod

from src.metrics import Metrics


class Gladiator(ABC):
    def __init__(self, number_of_epochs: int, metrics: Metrics, *args, **kwargs):
        self.number_of_epochs = number_of_epochs
        self.metrics = metrics
        self.weight = kwargs.get('default_neuron_weight', 0.1)
        self.learning_rate = kwargs.get('default_learning_rate', 0.001)

    @abstractmethod
    def train(self, training_data):
        pass
