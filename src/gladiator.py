from abc import ABC, abstractmethod


class Gladiator(ABC):

    def __init__(self, number_of_epochs):
        self.number_of_epochs = number_of_epochs

    @abstractmethod
    def train(self, training_data):
        pass
