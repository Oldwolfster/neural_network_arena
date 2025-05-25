class TrainingBatchInfo:
    def __init__(self, gladiators, arenas, lr_sweeps, dictionary_of_everything_else):
        self.gladiators = gladiators
        self.arenas = arenas
        self.lr_sweeps = lr_sweeps
        self.learning_rates = [1.0, 0.1, 0.01]
