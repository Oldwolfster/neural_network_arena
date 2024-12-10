from typing import List

from src.ArenaSettings import HyperParameters
from src.engine.MetricsMgr import MetricsMgr
from src.engine.convergence.Signal_UnderMeanThreshold_ShortTerm import Signal_UnderMeanThreshold_ShortTerm


class ConvergenceDetector:
    def __init__(self, hyper: HyperParameters, mgr: MetricsMgr, ):
        """
        Args:
            hyper (HyperParameters) : Stores hyperparameter configurations.
            mgr (MetricsMgr)        : Manages and tracks metrics across epochs.
        """
        self.hyper = hyper
        self.mgr = mgr
        self.relative_threshold = self.calculate_relative_threshold()
        self.signals = self.create_signals()
        self.triggered_signals = []

    def create_signals(self):
        return [
            Signal_UnderMeanThreshold_ShortTerm(self.mgr, self.relative_threshold)
            #,Signal_UnderMeanThreshold_LongTerm (self.mgr, self.relative_threshold)
        ]

    def check_convergence(self) -> bool:
        """
        Evaluate all signals - for now, if all are true we call it converged.
        Returns:
            List[str]: Signal Names that triggered convergence
        """
        self.mgr.convergence_signal.clear()  #Clear list - it should be empty, but better safe
        for signal in self.signals:
            triggered_signal = signal.evaluate()
            if triggered_signal:
                self.mgr.convergence_signal.append(triggered_signal)

        return len(self.mgr.convergence_signal) > 0


    def calculate_relative_threshold(self) -> float : # Compute the mean-based threshold for convergence using TrainingData.
        """
        Compute the mean-based threshold for convergence.

        The threshold is scaled by `converge_threshold`, treated as a percentage.
        For example, a `converge_threshold` of 5 means 5% of the mean target.

        Returns:
            float: The calculated convergence threshold.
        """
        td = self.mgr.training_data

        print(f"td.sum_targets={td.sum_of_targets}\ttd.sample_count={td.sample_count}")
        mean_of_targets = td.sum_of_targets / td.sample_count
        return mean_of_targets * self.hyper.converge_threshold / 100 # 100 makes the hyperparameter act as a %
