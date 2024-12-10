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
        self.epochs_within_threshold = 0
        self.has_converged = False

    def check_convergence(self, current_loss: float) -> int:
        """
        Check if the model has converged based on the current epoch's loss.

        :param current_loss: The loss value for the current epoch
        :return: Number of extra epochs since convergence was first detected, 0 otherwise
        """
        if self.has_converged:
            return self.epochs_within_threshold  # Return the number of epochs since convergence was detected

        # Initialize previous loss if it's the first epoch
        if self.previous_loss is None:
            self.previous_loss = current_loss
            return 0 # Convergence not yet reached; return zero

        #epsilon = 1e-10  # A small number to prevent division by zero
        #print(f'Current Loss{current_loss}')

        # Calculate percentage change for loss
        #if abs(self.previous_loss) > epsilon:
        if self.previous_loss == 0:
            loss_percentage_change = 0 # Not perfect but might just work
        else:
            loss_percentage_change = abs(self.previous_loss - current_loss) / abs(self.previous_loss)

        # Check if loss is less than the threshold
        if loss_percentage_change <= self.threshold_percentage: # or current_loss < .01:
            self.epochs_within_threshold += 1
        else:
            self.epochs_within_threshold = 0  # Reset counter if threshold is exceeded

        # Update previous value for next epoch
        self.previous_loss = current_loss

        # Check if convergence is reached
        if self.epochs_within_threshold >= self.required_epochs:
            self.has_converged = True
            return self.required_epochs
        return 0    # convergence was not reached so return zero to indicate such