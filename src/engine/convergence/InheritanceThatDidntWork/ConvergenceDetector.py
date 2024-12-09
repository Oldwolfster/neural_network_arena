from src.ArenaSettings import HyperParameters
from src.engine import MetricsMgr
from src.engine.convergence.Signal_MAEStabilization import MAEStabilizationSignal

class ConvergenceDetector:
    def __init__(self, hyper: HyperParameters, mgr: MetricsMgr, first_time = True):
        """
        Initialize the ConvergenceDetector.
        """
        self.hyper = hyper
        self.mgr = mgr
        if first_time:
            self.signals = self.create_signals()
        self.converged_signal = None
        self.extra_epochs = None

    def create_signals(self):


        return [
            MAEStabilizationSignal(self.hyper, self.mgr)
            #,F1PlateauSignal
        ]

    def evaluate(self):
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def name(self):
        return self.__class__.__name__

    def check_convergence(self):
        converged = False  # assume not converged

        print(f"cur epoch{self.mgr.epoch_curr_number}")
        for signal in self.signals:
            if signal.evaluate():
                converged = True
        if converged:
            return 2  # TODO Change this to epochs to remove