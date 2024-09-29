class ConvergenceDetector:
    def __init__(self, threshold_percentage: float, required_epochs: int):
        """
        Initialize the ConvergenceDetector.

        :param threshold_percentage: The percentage threshold for considering convergence (e.g., 1.0 for 1%)
        :param required_epochs: The number of epochs that must be within the threshold to declare convergence
        """
        self.threshold_percentage = threshold_percentage / 100.0  # Convert to decimal
        if required_epochs < 1:
            raise ValueError("Required Epochs must be one or higher")

        self.required_epochs = required_epochs

        # Instance variables (attributes)
        self.previous_loss = None
        self.previous_weight = None
        self.epochs_within_threshold = 0
        self.has_converged = False

    def check_convergence(self, current_loss: float, current_weight: float) -> int:
        """
        Check if the model has converged based on the current epoch's loss and weight.

        :param current_loss: The loss value for the current epoch
        :param current_weight: The weight value for the current epoch
        :return: Number of epochs to remove from log if converged, -1 otherwise
        """
        if self.has_converged:
            return self.required_epochs

        # Initialize previous loss and weight if it's the first epoch
        if self.previous_loss is None or self.previous_weight is None:
            self.previous_loss = current_loss
            self.previous_weight = current_weight
            return 0

        # Calculate percentage change for both loss and weight
        loss_percentage_change = abs(self.previous_loss - current_loss) / self.previous_loss
        weight_percentage_change = abs(self.previous_weight - current_weight) / self.previous_weight

        # Debugging output (can be removed)
        #print(f"Loss change: {loss_percentage_change}, Weight change: {weight_percentage_change}")

        # Check if both loss and weight are within the threshold
        if loss_percentage_change <= self.threshold_percentage and weight_percentage_change <= self.threshold_percentage:
            self.epochs_within_threshold += 1
        else:
            self.epochs_within_threshold = 0  # Reset counter if threshold is exceeded

        # Update previous values for next epoch
        self.previous_loss = current_loss
        self.previous_weight = current_weight

        # Check if convergence is reached
        if self.epochs_within_threshold >= self.required_epochs:
            self.has_converged = True
            return self.required_epochs

        return 0
