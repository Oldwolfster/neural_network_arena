from abc import ABC, abstractmethod
from src.engine.Metrics import GladiatorOutput, IterationResult, IterationContext
from src.engine.MetricsMgr import MetricsMgr
import numpy as np
from numpy import ndarray
from typing import Any


class Gladiator(ABC):
    def __init__(self, *args):
        gladiator = args[0]
        self.hyper = args[1]
        self.number_of_epochs = self.hyper.epochs_to_run
        self.metrics_mgr =  MetricsMgr(gladiator,  self.hyper)          # Create a new Metrics instance with the name as a string)  # Create a new Metrics instance with the name as a string
        #self.weight = self.hyper.default_neuron_weight
        self.weights = None  # Initialize to None      (or don't define yet)
        self.learning_rate = self.hyper.default_learning_rate
        self.bias = 0

    def train(self, training_data: list[tuple[float, ...]]) -> MetricsMgr:
        self.metrics_mgr.sample_count =  len(training_data)
        input_count = len(training_data[0])-1                           # get count of inputs (last element is target)

        if self.weights is None:                                         # If weights are not already initialized
            self.weights = np.full(input_count,                          # Now initialize self.weight as a NumPy array with `input_count` elements
                            self.hyper.default_neuron_weight)

        for epoch in range(self.number_of_epochs):                      # Loop to run specified # of epochs
            if self.run_an_epoch(training_data, epoch):                 # Call function to run single epoch
                return self.metrics_mgr                                 # Converged so end early
        return self.metrics_mgr                                         # When it does not converge still return metrics mgr

    def run_an_epoch(self, training_data: list[tuple[float, ...]], epoch_num: int) -> bool:         # Function to run single epoch
        for i, sample in enumerate(training_data):         # Loop through all the training data
            sample = np.array(sample)  # Convert sample to a NumPy array TODO Have training data be a numpy array to begin with

            # Record data for this iteration before passing sample to model
            context = IterationContext(
                iteration           = i + 1,
                epoch               = epoch_num + 1,
                inputs              = sample[:-1],          # All elements except the last
                target              = sample[-1],           # Last element as target
                weights             = np.copy(self.weights),# Weights PRIOR to model processing Sample
                bias                = self.bias,            # Bias    PRIOR to model processing Sample
                new_weights         = None,
                new_bias            = None
            )

            # Record model's prediction
            prediction              = self.training_iteration(sample)  # HERE IS WHERE IT PASSES CONTROL TO THE MODEL BEING TESTED
            gladiator_output        = GladiatorOutput(
                prediction          = prediction
            )

            # Put all the information together
            context.new_weights     = np.copy(self.weights) # Weights AFTER model processed sample
            context.new_bias        = self.bias             # Bias    AFTER model processed sample
            result = IterationResult(
                gladiator_output=gladiator_output,
                context=context
            )
            self.metrics_mgr.record_iteration(result)
        return self.metrics_mgr.finish_epoch_summary()


    @abstractmethod
    #def training_iteration(self, training_data: ndarray[Any, np.float64]) -> GladiatorOutput:
    def training_iteration(self, training_data: ndarray[Any, np.float64]) -> float:
        pass
