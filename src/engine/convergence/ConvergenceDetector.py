from typing import List

from src.ArenaSettings import HyperParameters
from src.engine.TrainingData import TrainingData
from src.engine.convergence.Signal_StableAccuracy import Signal_StableAccuracy
from src.engine.convergence.Signal_UnderMeanThreshold_ShortTerm import Signal_UnderMeanThreshold_ShortTerm





class ConvergenceDetector:
    def __init__(self, hyper: HyperParameters, td: TrainingData):
        """
        Args:
            hyper (HyperParameters) : Stores hyperparameter configurations.
            mgr (MetricsMgr)        : Manages and tracks metrics across epochs.
        """
        self.hyper = hyper
        self.td = td
        #self.mgr = mgr
        self.relative_threshold = self.calculate_relative_threshold()
        print(f"ConvergenceDetector - relative_threshold = {self.relative_threshold} ")
        self.metrics = []
        self.triggered_signals = []
        self.signals = self.create_signals()


    """
        @property
        def signals(self):
            if self._signals is None:
                # Lazy import to avoid circular reference
                self.create_signals()
            return self._converge_detector
    """
    def create_signals(self):
        return [
            Signal_UnderMeanThreshold_ShortTerm( self.relative_threshold, self.metrics),
            Signal_StableAccuracy(self.hyper.accuracy_threshold, self.metrics)
            #,Signal_UnderMeanThreshold_LongTerm (self.mgr, self.relative_threshold)
        ]

    def check_convergence(self, epoch_metrics : dict[str, float]) -> str:
        """
        Evaluate all signals - for now, if all are true we call it converged.
        Returns:
            List[str]: Signal Names that triggered convergence
        """

        self.metrics.append(epoch_metrics)
        for signal in self.signals:
            triggered_signal = signal.evaluate()
            if triggered_signal:
                self.triggered_signals.append(triggered_signal)

        return ", ".join(self.triggered_signals) if self.triggered_signals else ""

    def calculate_relative_threshold(self) -> float : # Compute the mean-based threshold for convergence using TrainingData.
        """
        Compute the mean-based threshold for convergence.

        The threshold is scaled by `converge_threshold`, treated as a percentage.
        For example, a `converge_threshold` of 5 means 5% of the mean target.

        Returns:
            float: The calculated convergence threshold.
        """
        #TODO only calculate this onece!

        print(f"CONVERGENCE DETECTOR td.sum_targets={self.td.sum_of_targets}\ttd.sample_count={self.td.sample_count}")
        mean_of_targets = self.td.sum_of_targets / self.td.sample_count
        rel_threshold = mean_of_targets * self.hyper.converge_threshold / 100 # 100 makes the hyperparameter act as a %
        return rel_threshold
