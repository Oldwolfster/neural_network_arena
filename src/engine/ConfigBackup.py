from src.Legos.ActivationFunctions import *
from .RamDB import RamDB
from .TrainingData import TrainingData
from src.Legos.WeightInitializers import *
from src.Legos.Optimizers import *
from .convergence.ConvergenceDetector import ROI_Mode
from ..ArenaSettings import HyperParameters
from ..Legos.LossFunctions import *
from dataclasses import dataclass, field

@dataclass
class Config:
    # 🔹 Shared components for all models
    hyper: HyperParameters                  = field(default_factory=HyperParameters)
    db: RamDB                               = field(default_factory=RamDB)
    training_data: TrainingData             = None
    lowest_error: float                     = 1e50
    lowest_error_epoch                      = 0
    backprop_headers                        = ["Config", "(*)", "Accp Blm", "=", "Raw Adj","LR", "=", "Final Adj"]
                                                #chg config to Input

    # 🔹 Unique components
    gladiator_name: str                     = ""
    optimizer: Optimizer                    = Optimizer_SGD
    batch_mode: BatchMode                   = BatchMode.SINGLE_SAMPLE
    batch_size: int                         = 2
    architecture: list                      = field(default_factory=lambda: [1])
    full_architecture: list                 = field(default_factory=lambda: [1,1])
    initializer: type                       = Initializer_Xavier

    loss_function: LossFunction             = Loss_MSE  # Default to MSE for regression and BCE For BD
    hidden_activation: type    = None  # Default to Tanh for regression and ReLU for BD
    output_activation: type    = None  # Will default to loss_function.recommended_output_activation if None
    roi_mode                                = ROI_Mode.SWEET_SPOT
    seconds: float                          = 0.0
    cvg_condition: str                      = "None"
    final_epoch: int                        =   0 # Last epoch to run
    popup_headers                           = None #TODO Standardize these 4 names.
    popup_operators                         = None
    popup_finalizer_headers                       = None
    popup_finalizer_operators                     = None




    def set_defaults(self):
        self.loss_function      = self.suggest_loss_function()
        self.hidden_activation  = self.suggest_activation_hidden()
        self.initializer        = self.suggest_initializer()
        self.output_activation  = self.loss_function.recommended_output_activation

    def configure_optimizer(self):
        (self.popup_headers, self.popup_operators,
         self.popup_finalizer_headers, self.popup_finalizer_operators) = self.optimizer.configure_optimizer(self)


    def suggest_loss_function(self) -> LossFunction:
        if self.training_data.problem_type == "Binary Decision":
            return Loss_BinaryCrossEntropy
        else:
            return Loss_MSE

    def suggest_activation_hidden(self):
        # Base on the loss and output needs
        if self.loss_function == Loss_BinaryCrossEntropy:
            return Activation_ReLU
        elif self.loss_function == Loss_MSE:
            return Activation_Tanh
        # add more...


    """
    def suggest_activation_output(self):
        # Base on the loss and output needs
        if self.loss_function == Loss_BinaryCrossEntropy:
            return Activation_Sigmoid
        elif self.loss_function == Loss_Hinge:
            return Activation_Tanh
        else:
            return Activation_NoDamnFunction
        # add more...
    """

    def suggest_initializer(self):
        if self.hidden_activation in [Activation_LeakyReLU, Activation_ReLU]:
            return Initializer_He
        if self.hidden_activation in [Activation_Sigmoid, Activation_Tanh]:
            return Initializer_Xavier
        return Initializer_Normal

    def suggest_optimizer(self) -> Optimizer:
        if self.training_data.problem_type == "Regression":
            return Optimizer_Adam
        return Optimizer_SGD

