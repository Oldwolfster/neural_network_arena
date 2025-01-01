from typing import Optional, List
from src.engine.convergence.Signal__BASE import Signal__BASE


class Signal_UnderMeanThreshold_ShortTerm(Signal__BASE):
    """
    Checks if the MAE of this epoch has improved less than the threshold since the previosu epoch
    """
    def __init__(self,  threshold: float , MAE_per_epoch):
        super().__init__(threshold,MAE_per_epoch)

    def evaluate(self, MAE_per_epoch : List[float]) -> Optional[str]:
        """
        Evaluates the signal and returns its name if it triggers, otherwise None.

        Returns:
            Optional[str]: The name of the signal if it evaluates to True, otherwise None.
        """
        if self.evaluate_mae_change(n_epochs=2):
            return self.signal_name
        return None

