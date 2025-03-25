from typing import Optional, List, Dict
from src.engine.convergence.Signal__BASE import Signal__BASE

class Signal_SweetSpot(Signal__BASE):
    """
    Triggers when the improvement in MAE is below a percentage threshold between consecutive epochs.
    """
    def __init__(self, threshold: float, metrics):
        super().__init__(threshold, metrics)

    def evaluate(self) -> Optional[str]:
        if len(self.metrics) < 2:
            return None
        mae_now = self.metrics[-1]['mean_absolute_error']
        mae_prev = self.metrics[-2]['mean_absolute_error']
        if abs(mae_now - mae_prev) < mae_now * self.threshold:
            return self.signal_name
        return None
