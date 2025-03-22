from src.engine.convergence.Signal__BASE import Signal__BASE
from typing import Optional

class Signal_StableAccuracy(Signal__BASE):
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
        if len(self.metrics) < n_epochs:
            return None  # Not enough data yet

        # Calculate the range (max - min) of accuracy in the last n_epochs
        recent_accuracies = [entry['Accuracy'] for entry in self.metrics[-n_epochs:]]
        stability = max(recent_accuracies) - min(recent_accuracies)

        # If the range is below the threshold, declare the signal triggered
        print(f"min acc = {min(recent_accuracies)}\tmax acc   = {max(recent_accuracies)}\tstability={stability}\tthreshold={self.threshold}")
        if stability < self.threshold:
            return self.signal_name
        return None
