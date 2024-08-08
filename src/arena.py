from abc import ABC, abstractmethod
from typing import List, Tuple, Any


class Arena(ABC):
    """
    Abstract base class for different types of training data arenas.
    """


    @abstractmethod
    def generate_training_data(self) -> List[Tuple[Any, ...]]:

        pass
