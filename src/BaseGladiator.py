from abc import ABC, abstractmethod
from src.Metrics import Metrics, GladiatorOutput, IterationResult, IterationContext
from src.MetricsMgr import MetricsMgr
from ArenaSettings import HyperParameters

class Gladiator(ABC):
    def __init__(self, *args):
        gladiator = args[0]
        hyper = args[1]
        self.number_of_epochs = hyper.epochs_to_run
        self.metrics_mgr =  MetricsMgr(gladiator,  hyper)  # Create a new Metrics instance with the name as a string)  # Create a new Metrics instance with the name as a string
        #self.weight = args[0]
        #self.learning_rate = args[1]
        self.weight = hyper.default_neuron_weight
        self.learning_rate = hyper.default_learning_rate

    def train(self, training_data: list[tuple[float, ...]]) -> MetricsMgr:
        for epoch in range(self.number_of_epochs):                      # Loop to run specified # of epochs
            if self.run_an_epoch(training_data, epoch):                 # Call function to run single epoch
                return self.metrics_mgr                                 # Converged so end early
        return self.metrics_mgr                                         # When it does not converge still return metrics mgr

    def run_an_epoch(self, training_data: list[tuple[float, ...]], epoch_num: int) -> bool:         # Function to run single epoch
        for i, sample in enumerate(training_data):         # Loop through all the training data
            gladiator_output = self.training_iteration(sample) # HERE IS WHERE IT PASSES CONTROL TO THE MODEL BEING TESTED
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
        return self.metrics_mgr.finish_epoch_summary()


    @abstractmethod
    def training_iteration(self, training_data: tuple[float]) -> GladiatorOutput:
        pass
