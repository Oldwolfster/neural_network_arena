class Arena_OffsetData(BaseArena):
    def __init__(self, num_samples: int = 20640):
        self.num_samples = num_samples
    def generate_training_data(self):
        samples = [
            (50, 0, 14000),
            (55, 0, 20000),
            (60, 0, 26000),
        ]
        return samples, ["Offset Feature", "Zero Feature", "Target"]
