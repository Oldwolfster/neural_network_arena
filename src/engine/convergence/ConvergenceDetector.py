from typing import List

from src.ArenaSettings import HyperParameters
from src.engine.RamDB import RamDB
from src.engine.TrainingData import TrainingData

from src.engine.convergence.Signal_PerfectAccuracy import Signal_PerfectAccuracy
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

    #TODO VERY INTERESING APPROACH.  Add as signal Normalized Convergence Rate Convergence_Rate = (previous_stability - current_stability) / (1 - current_stability)
    #TODO Oscillation detection Oscillation_Index = variance_of_sign_changes_in_recent_updates / variance_of_magnitude_changes
    def create_signals(self):
        return [
            #Signal_UnderMeanThreshold_ShortTerm( self.relative_threshold, self.metrics),
            #Signal_StableAccuracy(self.hyper.accuracy_threshold, self.metrics),
            Signal_PerfectAccuracy (self.hyper.accuracy_threshold, self.metrics),
            #Need to check every itSignal_GradientExplosion(self.hyper.accuracy_threshold, self.metrics)
            #,Signal_UnderMeanThreshold_LongTerm (self.mgr, self.relative_threshold)
        ]

    def check_convergence(self,epoch_current_no: int, epoch_metrics : dict[str, float]) -> str:
        """
        Evaluate all signals - for now, if all are true we call it converged.
        Returns:
            List[str]: Signal Names that triggered convergence
        """
        #TODO use ramdb!!!!   get_iteration_dict is already below!!!!
        #print(f"epoch metrics={epoch_metrics}")
        self.metrics.append(epoch_metrics)

        #print(f"epoch metrics={self.metrics[-1]}")
        if len(self.metrics) < self.hyper.min_no_epochs:
            return ""   # Has not yet met minimum no of epochs per hyper paramater setting

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

    def get_iteration_dict(self, db: RamDB, epoch: int, iteration: int) -> dict:  #Retrieve iteration data from the database."""
        # db.query_print("PRAGMA table_info(Iteration);")
        sql = """  
            SELECT * FROM Iteration 
            WHERE epoch = ? AND iteration = ?  
        """#TODO ADD MODEL TO CRITERIIA
        params = (epoch, iteration)
        rs = db.query(sql, params)

        if rs:            #
            return rs[0]  # Return the first row as a dictionary

        #print(f"No data found for epoch={epoch}, iteration={iteration}")
        return {}  # Return an empty dictionary if no results

