from typing import Optional, List, Dict
from src.NNA.engine.convergence.Signal__BASE import Signal__BASE

class Signal_Economic(Signal__BASE):
    """
    Triggers when the improvement in MAE is below a percentage threshold between consecutive epochs.
    """
    def __init__(self, threshold: float, metrics):
        super().__init__(threshold, metrics)

    def evaluate(self) -> Optional[str]:
        if len(self.metrics) < 3:
            return None
        mae_now     = self.metrics[-1]['mean_absolute_error']
        mae_prev1   = self.metrics[-2]['mean_absolute_error']
        mae_prev2   = self.metrics[-3]['mean_absolute_error']
        prev_drop   = mae_prev2 - mae_prev1
        recent_drop = mae_prev1 - mae_now
        #print(self.metrics[-1]['epoch'])
        if prev_drop > 0 and recent_drop < prev_drop * self.threshold:
            return self.signal_name
        return None

