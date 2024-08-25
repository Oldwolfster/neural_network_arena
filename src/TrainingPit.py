from abc import ABC, abstractmethod
from typing import List, Tuple, Any


class TrainingPit(ABC):
    """
    Abstract base class for different types of training data TrainingPits.
    """


    @abstractmethod
    def generate_training_data(self) -> List[Tuple[Any, ...]]:

        pass
