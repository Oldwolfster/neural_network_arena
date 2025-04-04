from abc import ABC, abstractmethod
from typing import Optional, List

class Signal__BASE(ABC):
    """
    Base class for all convergence signals.

    This abstract class defines the interface for signal evaluation, ensuring
    all derived signal classes implement the `evaluate` method.

    Attributes:
        mgr (MetricsMgr): Provides access to metrics and data for evaluation.
        threshold (float): The threshold value used to evaluate convergence.
    """

    def __init__(self,  threshold: float, metrics):
        """
        Initialize the base signal.

        Args:
            metrics (dict[str, float]): The metrics manager, providing access to error metrics for current epoch
            threshold (float): The threshold value for evaluating convergence.
        """

        self.threshold = threshold
        self.metrics = metrics

    @property
    def signal_name(self):
        return self.__class__.__name__
    @abstractmethod
    def evaluate(self) -> Optional[str]:
        """
        Evaluate the signal for convergence.

        This method must be implemented in derived classes.

        Returns:
            str: Signal Name if true, otherwise None
        """
    def evaluate_mae_change(self, n_epochs: int, mae_threshold:  float) -> bool:
        """
        Evaluate if the percentage decrease in MAE over the last n_epochs is below the threshold.

        Args:
            n_epochs (int): Number of epochs to consider in rolling improvement calculation.

        Returns:
            bool: True if the MAE change is below the threshold (stall detected), otherwise False.
        """
        if len(self.metrics) < n_epochs:
            return False  # Not enough epochs to compare

        MAE_now = self.metrics[-1]['mean_absolute_error']
        MAE_prior = self.metrics[-n_epochs]['mean_absolute_error']

        if MAE_prior == 0:  # Prevent division by zero
            return False

        improvement_rate = (MAE_prior - MAE_now) / MAE_prior  # Percentage decrease

        # print(f"MAE_now={MAE_now}, MAE_prior={MAE_prior}, Improvement Rate={improvement_rate}, Threshold={self.threshold}")

        return improvement_rate < mae_threshold  # Flag as stall if below threshold



    def evaluate_mae_changeAbs(self, n_epochs: int) -> bool:
        """
        Evaluate if the MAE change over the last n_epochs is below the threshold.

        Args:
            n_epochs (int): The number of epochs to consider for the evaluation.

        Returns:
            bool: True if the MAE change is below the threshold, otherwise False.
        """
        #if self.mgr.epoch_curr_number < n_epochs:  # Not enough epochs to compare
        if len(self.metrics) < n_epochs:   # Not enough epochs to compare
            return False

        MAE_now = self.metrics[-1]['mean_absolute_error']  # epoch_metrics['mean_absolute_error']}
        MAE_prior = self.metrics[-n_epochs]['mean_absolute_error']
        change = abs(MAE_now - MAE_prior)


        #print(f"Signal__BASE - change={change}\tthreshold={self.threshold} ")
        #          f"MAE_now={MAE_now}, MAE_prior={MAE_prior}, change={change}\tthreshold {self.threshold}")
        return change < self.threshold
