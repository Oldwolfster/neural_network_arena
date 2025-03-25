from typing import Optional
from src.engine.convergence.Signal__BASE import Signal__BASE


class Signal_CheckForRecovery(Signal__BASE):
    """
    Fires if the MAE slope has increased significantly compared to the prior N epochs.
    This can indicate that the model is recovering after a stall or decay phase.
    """

    def __init__(self, threshold: float, metrics):
        super().__init__(threshold, metrics)

    def evaluate(self) -> Optional[str]:
        if len(self.metrics) < 20:
            return None

        # Window for checking change in slope
        recent_window = self.metrics[-5:]
        prev_window = self.metrics[-10:-5]

        def get_slope(data):
            # Simple slope: difference over window length
            return (data[-1]['mean_absolute_error'] - data[0]['mean_absolute_error']) / len(data)

        recent_slope = get_slope(recent_window)
        prev_slope = get_slope(prev_window)

        # If slope is now significantly more negative (i.e., steeper drop in MAE), we say slope sharpened
        slope_improvement = prev_slope - recent_slope

        if slope_improvement > self.threshold:
            return self.signal_name

        return None
