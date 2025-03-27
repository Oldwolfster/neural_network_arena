from typing import Optional, List, Dict
from src.engine.Utils_DataClasses import ez_debug
from src.engine.convergence.Signal__BASE import Signal__BASE
from typing import Optional
from src.engine.convergence.Signal__BASE import Signal__BASE

class Signal_CornerCatch(Signal__BASE):
    """
    Triggers when improvement from previous epoch becomes small relative to the initial improvement.
    Uses a grace period (default: 5 epochs) before evaluating, to allow learning to stabilize.
    """

    def __init__(self, threshold: float, metrics: List[Dict], grace_period: int = 5):
        super().__init__(threshold, metrics)
        self.grace_period = grace_period

    def evaluate(self) -> Optional[str]:
        if len(self.metrics) < self.grace_period + 2:
            return None  # not enough data yet

        mae_now    = self.metrics[-1]['mean_absolute_error']
        mae_prev   = self.metrics[-2]['mean_absolute_error']
        mae_start  = self.metrics[self.grace_period]['mean_absolute_error']  # Baseline after grace

        baseline_drop = mae_start - self.metrics[self.grace_period + 1]['mean_absolute_error']
        current_drop  = mae_prev  - mae_now

        # Prevent divide-by-zero
        if baseline_drop <= 0:
            return None

        ratio = current_drop / baseline_drop

        # Debugging info
        ez_debug(
            mae_now=mae_now,
            mae_prev=mae_prev,
            mae_start=mae_start,
            baseline_drop=baseline_drop,
            current_drop=current_drop,
            ratio=ratio,
            threshold=self.threshold
        )

        if ratio < self.threshold:
            return self.signal_name
        return None

