from .ActivationFunction import *
from .RamDB import RamDB
from .SQL import retrieve_training_data

from .TrainingData import TrainingData
from .WeightInitializer import *
from ..ArenaSettings import run_previous_training_data, HyperParameters
from ..Legos.LossFunctions import *

from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    # 🔹 Hyperparameters and # 🔹 Database Connections
    hyper: HyperParameters      = field(default_factory=HyperParameters)
    db: RamDB                   =  field(default_factory=RamDB)
    training_data: TrainingData = None

    # 🔹 Architecture (Using Strategy Pattern for most)
    gladiator_name: str = ""
    architecture: list = field(default_factory=lambda: [1])
    initializer: type = Initializer_Xavier
    activation_function_for_hidden: type = Tanh
    loss_function: LossFunction = Loss_MSE  # Default to MSE  # 🔹 Loss Function
    optimizer: str = "simplified_descent"  # Default to your method


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


