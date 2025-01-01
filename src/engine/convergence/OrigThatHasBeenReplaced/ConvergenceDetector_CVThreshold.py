class ConvergenceDetector:
    def __init__(self, window_size=20, cv_threshold=0.002, required_windows=3, patience=2):
        """
        Initialize the ConvergenceDetector.

        :param window_size: Number of epochs in each window for calculating statistics
        :param cv_threshold: Threshold for the coefficient of variation to consider convergence
        :param required_windows: Number of consecutive windows meeting the criteria to declare convergence
        :param patience: Number of consecutive windows allowed to not meet the criteria before resetting
        """
        self.window_size = window_size
        self.cv_threshold = cv_threshold
        self.required_windows = required_windows
        self.patience = patience

        self.loss_history = []
        self.converged_windows = 0
        self.non_converged_windows = 0
        self.epochs_since_convergence = -1  # -1 indicates convergence not yet detected

    def check_convergence(self, current_loss):
        """
        Check if the model has converged based on the current epoch's loss.

        :param current_loss: The loss value for the current epoch
        :return: Number of epochs since convergence was detected, 0 if not converged
        """
        self.loss_history.append(current_loss)

        if len(self.loss_history) < self.window_size:
            return 0  # Not enough data yet to determine convergence

        # Calculate statistics over the current window
        current_window = self.loss_history[-self.window_size:]
        mean_loss = sum(current_window) / self.window_size
        variance_loss = sum((x - mean_loss) ** 2 for x in current_window) / self.window_size
        std_dev_loss = variance_loss ** 0.5

        # Compute coefficient of variation (CV)
        cv = std_dev_loss / mean_loss if mean_loss != 0 else 0.0

        # Check convergence
        if cv <= self.cv_threshold:
            self.converged_windows += 1
            self.non_converged_windows = 0  # Reset non-converged window count
            if self.converged_windows >= self.required_windows:
                if self.epochs_since_convergence == -1:
                    # Convergence just detected
                    self.epochs_since_convergence = 0
                else:
                    # Increment epochs since convergence was first detected
                    self.epochs_since_convergence += 1
                return self.epochs_since_convergence
            else:
                return 0
        else:
            self.non_converged_windows += 1
            if self.non_converged_windows > self.patience:
                # Reset counters if patience exceeded
                self.converged_windows = 0
                self.epochs_since_convergence = -1
                self.non_converged_windows = 0
            return 0
