class Arena_KillRelus(BaseArena):
    def __init__(self, num_samples: int):
    self.num_samples = num_samples

    def generate_training_data(self):
        self.description = "Intentionally triggers ReLU death"
        self.feature_labels = ["X1", "X2"]
        self.target_labels = ["Y"]

        self.data = [
            (-1.0, -1.0, 0.0),
            (-2.0, -3.0, 0.0),
            (-0.5, -0.2, 0.0),
            (-5.0, -1.0, 0.0),
            (-3.0, -4.0, 0.0),
            (-10.0, -10.0, 0.0),  # extra deadly
