from typing import List

from src.ArenaSettings import HyperParameters
from src.engine.RamDB import RamDB
from src.engine.TrainingData import TrainingData

from src.engine.convergence.Signal_PerfectAccuracy import Signal_PerfectAccuracy
from src.engine.convergence.DEAD_Signal_StableAccuracy import Signal_StableAccuracy
from src.engine.convergence.DEAD_Signal_UnderMeanThreshold_ShortTerm import Signal_UnderMeanThreshold_ShortTerm
from src.engine.convergence.Signal_RollingMaeImprovement10 import Signal_RollingMaeImprovement10
    #TODO VERY INTERESING APPROACH.  Add as signal Normalized Convergence Rate Convergence_Rate = (previous_stability - current_stability) / (1 - current_stability)
    #TODO Oscillation detection Oscillation_Index = variance_of_sign_changes_in_recent_updates / variance_of_magnitude_changes


class ConvergenceDetector:
    def __init__(self, hyper: HyperParameters, td: TrainingData):
        """
        Args:
            hyper (HyperParameters) : Stores hyperparameter configurations.
            mgr (MetricsMgr)        : Manages and tracks metrics across epochs.
        """
        self.hyper = hyper
        self.td = td
        #self.relative_threshold = self.calculate_relative_threshold()
        self.metrics = []
        self.triggered_signals = []
        self.signals = self.create_signals()

    def create_signals(self):
        return [
            Signal_PerfectAccuracy (self.hyper.accuracy_threshold, self.metrics),
            Signal_RollingMaeImprovement10(self.hyper.accuracy_threshold, self.metrics)
        ]

    def check_convergence(self,epoch_current_no: int, epoch_metrics : dict[str, float]) -> str:
        """
        Evaluate all signals - for now, if all are true we call it converged.
        Returns:
            List[str]: Signal Names that triggered convergence
        """

        #print(f"epoch metrics={epoch_metrics}")
        self.metrics.append(epoch_metrics)

        if len(self.metrics) < self.hyper.min_no_epochs:
            return ""   # Has not yet met minimum no of epochs per hyper paramater setting

        for signal in self.signals:
            triggered_signal = signal.evaluate()
            if triggered_signal:
                self.triggered_signals.append(triggered_signal)
        return ", ".join(self.triggered_signals) if self.triggered_signals else ""

    def get_iteration_dict(self, db: RamDB, epoch: int, iteration: int) -> dict:  #Retrieve iteration data from the database."""
        """
        NOT CURRENTLY IN USE BUT WE MAY NEED IT
        """
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
