class ConvergenceDetector:
    def __init__(self, min_delta: float, patience: int):
        self.min_delta = min_delta
        self.patience = patience
        self.best_loss = None
        self.epochs_no_improve = 0
        self.has_converged = False

    def check_convergence(self, current_loss: float) -> bool:
        if self.best_loss is None or current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            self.has_converged = True

        return self.has_converged
