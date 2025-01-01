from abc import ABC, abstractmethod
from typing import Optional, List

from src.engine.MetricsMgr import MetricsMgr



class Signal__BASE(ABC):
    """
    Base class for all convergence signals.

    This abstract class defines the interface for signal evaluation, ensuring
    all derived signal classes implement the `evaluate` method.

    Attributes:
        mgr (MetricsMgr): Provides access to metrics and data for evaluation.
        threshold (float): The threshold value used to evaluate convergence.
    """

    def __init__(self,  threshold: float, MAE_per_epoch: List[float]):
        """
        Initialize the base signal.

        Args:
            mgr (MetricsMgr): The metrics manager, providing access to error metrics.
            threshold (float): The threshold value for evaluating convergence.
        """

        self.threshold = threshold
        self.MAE_per_epoch = MAE_per_epoch

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

    def evaluate_mae_change(self, n_epochs: int) -> bool:
        """
        Evaluate if the MAE change over the last n_epochs is below the threshold.

        Args:
            n_epochs (int): The number of epochs to consider for the evaluation.

        Returns:
            bool: True if the MAE change is below the threshold, otherwise False.
        """
        #if self.mgr.epoch_curr_number < n_epochs:  # Not enough epochs to compare
        if len(self.MAE_per_epoch) < n_epochs:
            return False

        MAE_now = self.MAE_per_epoch[-1]
        MAE_prior = self.MAE_per_epoch[-n_epochs]
        change = abs(MAE_now - MAE_prior)

        #if self.mgr.epoch_curr_number % 100 == 0:
        #    print(f"{self.mgr.name} MAE over {n_epochs} epochs:{self.mgr.epoch_curr_number} "
        #          f"MAE_now={MAE_now}, MAE_prior={MAE_prior}, change={change}\tthreshold {self.threshold}")
        return change < self.threshold
