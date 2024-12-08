class ConvergenceDetector:
    def __init__(self, threshold_percentage: float, required_epochs: int, target_sum: float, dynamic_threshold_factor=0.001):
        self.threshold_percentage = threshold_percentage / 100.0  # Convert to decimal
        self.required_epochs = required_epochs
        self.target_sum = target_sum
        self.dynamic_threshold = target_sum * dynamic_threshold_factor  # Scaled threshold based on target sum
        self.previous_loss = None
        self.epochs_within_threshold = 0
        self.has_converged = False

    def check_convergence(self, current_loss: float) -> int:
        if self.has_converged:
            return self.required_epochs

        if self.previous_loss is None:
            self.previous_loss = current_loss
            return 0

        epsilon = 1e-10
        loss_percentage_change = abs(self.previous_loss - current_loss) / (abs(self.previous_loss) + epsilon)
        print("checking convergence")

        # Check if loss is within the dynamic threshold or percentage threshold
        if current_loss < self.dynamic_threshold or loss_percentage_change <= self.threshold_percentage:
            self.epochs_within_threshold += 1
        else:
            self.epochs_within_threshold = 0

        self.previous_loss = current_loss

        if self.epochs_within_threshold >= self.required_epochs:
            self.has_converged = True
            return self.required_epochs

        return 0
