from src.Legos.ActivationFunctions import *
from .RamDB import RamDB
from .TrainingData import TrainingData
from src.Legos.WeightInitializers import *
from .convergence.ConvergenceDetector import ROI_Mode
from ..ArenaSettings import HyperParameters
from ..Legos.LossFunctions import *
from dataclasses import dataclass, field

@dataclass
class Config:
    # 🔹 Shared components for all models
    hyper: HyperParameters         = field(default_factory=HyperParameters)
    db: RamDB                      = field(default_factory=RamDB)
    training_data: TrainingData    = None

    # 🔹 Unique components
    gladiator_name: str                     = ""
    architecture: list                      = field(default_factory=lambda: [1])
    full_architecture: list                 = field(default_factory=lambda: [1,1])
    initializer: type                       = Initializer_Xavier
    loss_function: LossFunction             = Loss_MSE  # Default to MSE for regression and BCE For BD
    hidden_activation: type    = None  # Default to Tanh for regression and ReLU for BD
    output_activation: type    = None  # Will default to loss_function.recommended_output_activation if None
    roi_mode                                = ROI_Mode.SWEET_SPOT
    seconds: float                          = 0.0
    cvg_condition: str                      = "None"

    def set_defaults(self):
        if self.training_data.problem_type == "Binary Decision":
            self.loss_function = Loss_BinaryCrossEntropy
            self.hidden_activation = Activation_ReLU
        else:
            self.loss_function = Loss_MSE
            self.hidden_activation = Activation_Tanh
        self.output_activation = self.loss_function.recommended_output_activation
        print (f"Set Default")
