from src.NNA.engine.convergence.Signal__BASE import Signal__BASE
from typing import Optional

class Signal_PerfectAccuracy(Signal__BASE):
    """
    Checks if the classification accuracy has stabilized over the past n_epochs.
    """
    def __init__(self, threshold: float, metrics):
        super().__init__(threshold, metrics)

    def evaluate(self) -> Optional[str]:
        """
        Evaluates the signal based on changes in accuracy over recent epochs.

        Args:
            metrics (List[dict]): The list of accuracy values per epoch.

        Returns:
            Optional[str]: The name of the signal if it evaluates to True, otherwise None.
        """
        n_epochs = 8  # Number of epochs to consider for stabilization

        #print(f"self.metrics.epoch_curr_number:{self.metrics[-1].epoch_curr_number}\tself.metrics[-1]={self.metrics[-1]}")
        if float(self.metrics[-1]['Accuracy'])   != 100:
            return None  # Not enough data yet

        return self.signal_name     #signal_name inherited from Signal__BASE

