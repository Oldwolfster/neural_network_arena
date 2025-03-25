from typing import Optional, List
from src.engine.convergence.Signal__BASE import Signal__BASE


class Signal_MostAccurate(Signal__BASE):
    """
    Checks if the MAE of this epoch has improved less than the threshold since the previosu epoch
    """
    def __init__(self,  threshold: float , metrics):
        super().__init__(threshold,metrics)
        #print(f"yo - setting threshold - {threshold}")

    def evaluate(self) -> Optional[str]:
        """
        Evaluates the signal and returns its name if it triggers, otherwise None.

        Returns:
            Optional[str]: The name of the signal if it evaluates to True, otherwise None.
        """
        #print(f"In RollingMAE{self.threshold}")
        if self.evaluate_mae_change(n_epochs= 20, mae_threshold=self.threshold): #larger number convg earlier
            return self.signal_name
        return None

