class ConvergenceDetector:
    def __init__(self, hyper: HyperParameters, mgr: MetricsMgr, training_data: TrainingData, first_time=True):
        """
        Initialize the ConvergenceDetector.
        """
        self.hyper = hyper
        self.mgr = mgr
        self.training_data = training_data  # Added reference to TrainingData
        self.signals = self.create_signals()
        self.relative_threshold = self.calculate_relative_threshold()

    def calculate_relative_threshold(self):
        """
        Compute the mean-based threshold for convergence using TrainingData.
        """
        mean_target = self.training_data.sum_of_targets / self.training_data.sample_count
        threshold = self.hyper.convergence_factor * mean_target  # Use a single factor to scale
        print(f"Mean-Based Threshold: {threshold}")
        return threshold
