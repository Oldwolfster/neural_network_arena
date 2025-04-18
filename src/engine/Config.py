from src.Legos.ActivationFunctions import *
from .RamDB import RamDB
from .TrainingData import TrainingData
from src.Legos.WeightInitializers import *
from src.Legos.Optimizers import *
from .convergence.ConvergenceDetector import ROI_Mode
from ..ArenaSettings import HyperParameters
from ..Legos.LegoSelector import LegoSelector
from ..Legos.LossFunctions import *
from dataclasses import dataclass, field

@dataclass
class Config:
    # 🔹 Shared components for all models
    hyper:          HyperParameters         = field(default_factory=HyperParameters, repr=False, compare=False, metadata={"asdict": False})
    lego_selector:  LegoSelector            = field(default_factory=LegoSelector, repr=False, compare=False, metadata={"asdict": False})
    #db:             RamDB                   = field(default_factory=RamDB)
    db:             RamDB                    = field(default_factory=RamDB, repr=False, compare=False, metadata={"asdict": False})
    training_data:  TrainingData            = None

    # 🔹 Unique components
    gladiator_name: str                     = ""
    optimizer: Optimizer                    = None
    batch_mode: BatchMode                   = None
    batch_size: int                         = 10
    architecture: list                      = field(default_factory=lambda: [1])
    initializer: type                       = None
    loss_function: LossFunction             = None
    hidden_activation: type                 = None  # Default to Tanh for regression and ReLU for BD
    output_activation: type                 = None  # Will default to loss_function.recommended_output_activation if None


    # Misc attributes
    full_architecture: list                 = field(default_factory=lambda: [1,1])
    roi_mode                                = ROI_Mode.SWEET_SPOT
    seconds: float                          = 0.0
    cvg_condition: str                      = "None"
    final_epoch: int                        =   0 # Last epoch to run
    lowest_error: float                     = 1e50
    lowest_error_epoch                      = 0
    backprop_headers                        = ["Config", "(*)", "Accp Blm", "=", "Raw Adj","LR", "=", "Final Adj"]

    popup_headers                           = None #TODO Standardize these 4 names.
    popup_operators                         = None
    popup_finalizer_headers                 = None
    popup_finalizer_operators               = None

    def configure_optimizer(self):
        (self.popup_headers, self.popup_operators,
         self.popup_finalizer_headers, self.popup_finalizer_operators) = self.optimizer.configure_optimizer(self)

    def set_defaults(self):
        #self.lego_selector = LegoSelector(self)
        self.smartNetworkSetup()

    def smartNetworkSetup(self):
        self.lego_selector.apply(self, self.get_rules())

    def get_rules(self):        #  config.training_data.problem_type == " (0, 100, {"loss_function": Loss_BCEWithLogits}, "config.training_data.problem_type == 'Binary Decision'"),
        return [
            (0, 100, {"output_activation": Activation_Sigmoid}, "training_data.problem_type == 'Binary Decision'"),
            (0, 100, {"output_activation": Activation_NoDamnFunction}, "training_data.problem_type != 'Binary Decision'"),
            (0, 200, {"loss_function": Loss_BCEWithLogits}, "output_activation.name == 'Sigmoid'"),
            #(0, 210, {"loss_function": Loss_MSE}, "training_data.has_high_outliers()"),
            #(0, 210, {"loss_function": Loss_MAE}, "not training_data.has_high_outliers()"),

            (0, 669, {"optimizer": Optimizer_Adam}, "1 == 1"),
            (0, 669, {"batch_mode": BatchMode.MINI_BATCH}, "1 == 1"),
            (0, 669, {"batch_size": 1}, "1 == 1"),
            (0, 669, {"architecture": [1]}, "1 == 1"),
            (0, 669, {"initializer": Initializer_Xavier}, "1 == 1"),
            (0, 669, {"loss_function": Loss_MAE}, "1 == 1"),
            (0, 669, {"hidden_activation": Activation_Tanh}, "1 == 1"),
            (0, 669, {"output_activation": Activation_NoDamnFunction}, "1 == 1"),
        ]


