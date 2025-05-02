from enum import Enum
from typing import List, Dict
from src.ArenaSettings import HyperParameters
from src.engine.RamDB import RamDB
from src.engine.TrainingData import TrainingData
from src.engine.convergence.Signal_CornerCatcher import Signal_CornerCatch
from src.engine.convergence.Signal_Economic import Signal_Economic

from src.engine.convergence.Signal_PerfectAccuracy import Signal_PerfectAccuracy
from src.engine.convergence.Signal_MostAccurate import Signal_MostAccurate
from src.engine.convergence.Signal_SweetSpot import Signal_SweetSpot


class ROI_Mode(Enum):
    ECONOMIC            = "economic"  # Stop early, only major gains
    SWEET_SPOT          = "sweet_spot"  # Default — stop when gains plateau
    MOST_ACCURATE       = "most_accurate"  # Squeeze every drop

class ConvergenceDetector:
    def __init__(self, hyper: HyperParameters, td: TrainingData, config):
        self.hyper = hyper
        self.td = td
        self.metrics = []
        self.triggered_signals: List[str] = []
        self.phase = "watch"        # Phase state: 'watch', 'fix', 'done'

        #print(f"{config.roi_mode}")

        # Map of phases to signal classes
        self.phase_signals = {
            "watch": [
                Signal_PerfectAccuracy(self.hyper.accuracy_threshold, self.metrics)
                #,Signal_CornerCatch(.1, self.metrics)
               #,self.get_roi_signal(config.roi_mode, self.hyper, self.metrics)
            ],
            "fix": [
                # Placeholder, no fix signals active yet
            ],
            "done": [
            ]
        }

        # Track signals that have fired in each phase
        self.phase_signals_fired: Dict[str, List[str]] = {
            "watch": [],
            "fix": [],
            "done": []
        }

    def check_convergence(self, epoch_current_no: int, epoch_metrics: dict[str, float]) -> str:
        """
        Evaluate signals for the current phase.
        Returns:
            str: "" to continue training, or name of signal (or sentinel) to trigger behavior
        """
        self.metrics.append(epoch_metrics)

        if len(self.metrics) < self.hyper.min_no_epochs:
            return ""

        signals_for_current_phase = self.phase_signals.get(self.phase, [])

        for signal in signals_for_current_phase:
            result = signal.evaluate()
            if result:
                self.phase_signals_fired[self.phase].append(result)

        # === TEMP: Preserve old behavior ===
        # Fire convergence immediately if any signal fired
        if self.phase_signals_fired[self.phase]:
            #if self.phase == "watch":
            #    self.phase = "fix"
            #    return self.phase
            return ", ".join(self.phase_signals_fired[self.phase])

        return "Did Not Converge"

    def get_iteration_dict(self, db: RamDB, epoch: int, iteration: int) -> dict:
        sql = """  
            SELECT * FROM Iteration 
            WHERE epoch = ? AND iteration = ?  
        """
        params = (epoch, iteration)
        rs = db.query(sql, params)
        return rs[0] if rs else {}

    def get_roi_signal(self, roi_mode: ROI_Mode, hyper, metrics):
        print (f"roi_mode={roi_mode}")
        if roi_mode == ROI_Mode.MOST_ACCURATE:
            return Signal_MostAccurate(hyper.threshold_Signal_MostAccurate, metrics)
        elif roi_mode == ROI_Mode.SWEET_SPOT:
            return Signal_SweetSpot(hyper.threshold_Signal_SweetSpot, metrics)
        elif roi_mode == ROI_Mode.ECONOMIC:
            return Signal_Economic(hyper.threshold_Signal_Economic, metrics)
        else:
            raise ValueError(f"Unsupported ROI mode: {roi_mode}")

