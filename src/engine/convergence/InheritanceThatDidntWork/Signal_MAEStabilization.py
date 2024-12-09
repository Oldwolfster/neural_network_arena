


class MAEStabilizationSignal():
    def __init__(self, hyper, mgr):
        #super().__init__(hyperparameters, mgr, False)
        self.hyper = hyper
        self.mgr = mgr

    def evaluate(self):
        if self.mgr.epoch_curr_number > 5:
            return True
        return False

