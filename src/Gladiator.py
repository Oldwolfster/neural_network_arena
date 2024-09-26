from abc import ABC, abstractmethod
from src.Metrics import Metrics, GladiatorOutput, IterationResult, IterationContext
from src.MetricsMgr import MetricsMgr

class Gladiator(ABC):
    def __init__(self, number_of_epochs: int, metrics_mgr: MetricsMgr, *args):
        self.number_of_epochs = number_of_epochs
        self.metrics_mgr = metrics_mgr
        self.weight = args[0]
        self.learning_rate = args[1]

    def train(self, training_data: list[tuple[float, ...]]) -> None:
        for epoch in range(self.number_of_epochs):                      # Loop to run specified # of epochs
            if self.run_an_epoch(training_data, epoch):                 # Call function to run single epoch
                break

    def run_an_epoch(self, training_data: list[tuple[float, ...]], epoch_num: int) -> bool:         # Function to run single epoch
        for i, sample in enumerate(training_data):         # Loop through all the training data
            gladiator_output = self.training_iteration(sample) # HEE IS WHERE IT PASSES CONTROL TO THE MODEL BEING TESTED
            context = IterationContext(
                iteration=i + 1,
                epoch=epoch_num + 1,
                input=sample[0],
                target=sample[1]
            )

            result = IterationResult(
                gladiator_output=gladiator_output,
                context=context
            )
            self.metrics_mgr.record_iteration(result)
        return self.metrics_mgr.check_for_convergence()                              # Sends back the data for an epoch

    @abstractmethod
    def training_iteration(self, training_data: tuple[float]) -> GladiatorOutput:
        pass
