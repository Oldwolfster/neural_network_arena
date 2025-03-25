from src.Legos.ActivationFunctions import *
from .RamDB import RamDB

from .TrainingData import TrainingData
from src.Legos.WeightInitializers import *
from .convergence.ConvergenceDetector import ROI_Mode

from ..ArenaSettings import HyperParameters
from ..Legos.LossFunctions import *

from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    # 🔹 Shared components for all models
    hyper: HyperParameters      = field(default_factory=HyperParameters)
    db: RamDB                   =  field(default_factory=RamDB)
    training_data: TrainingData = None

    # 🔹 Unique components
    gladiator_name: str                     = ""
    architecture: list                      = field(default_factory=lambda: [1])
    full_architecture: list                 = field(default_factory=lambda: [1])
    initializer: type                       = Initializer_Xavier
    loss_function: LossFunction             = Loss_MSE  # Default to MSE for regression and BCE For BD # 🔹 Loss Function
    activation_function_for_hidden: type    = Activation_Tanh # Default to Tanh for regression and Relu for BD
    roi_mode                                = ROI_Mode.SWEET_SPOT
    seconds: float                          = 0.0
    cvg_condition: str                      = ""

    def set_defaults(self):
        if self.training_data.problem_type      == "Binary Decision":
            self.loss_function                  = Loss_BinaryCrossEntropy
            self.activation_function_for_hidden = Activation_ReLU # Default to Tanh for regression and Relu for BD

#    def __post_init__(self):
#        """ Initialize training_data AFTER hyper is available. """
#        self.training_data = get_training_data(self.hyper)  # ✅ Now correctly gets training data

    ### Below here has yet to be migrated
    """
    # 🔹 Core Training Settings
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 1  # Defaults to SGD
    # 🔹 Hyperparameters
    accuracy_threshold: float = 0.01
    momentum: float = 0.9    
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32
    regularization: Optional[str] = None
    momentum: float = 0.9
    early_stopping: Optional[Dict[str, Any]] = None
    metrics: List[str] = ('accuracy',)
    verbose: bool = True
    training_data: Optional[Any] = None
    validation_data: Optional[Any] = None
    database_connection: Optional[Any] = None
    """


