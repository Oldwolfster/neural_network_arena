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
    optimizer                               = None #: Optimizer
    batch_mode: BatchMode                   = None
    batch_size: int                         = None
    architecture: list                      = field(default_factory=lambda: [1])
    initializer: type                       = None
    loss_function: StrategyLossFunction             = None
    hidden_activation: type                 = None  # Default to Tanh for regression and ReLU for BD
    output_activation: type                 = None  # Will default to loss_function.recommended_output_activation if None
    input_scaler                            = None
    roi_mode                                = ROI_Mode.SWEET_SPOT

    # Misc attributes  ##MAKE SURE TO ADD TO DATA RESET
    seconds: float                          = 0.0
    cvg_condition: str                      = "Did Not Converge"
    learning_rate: float                       = 0.0       # Read in beginning to instantiate  neurons with correct LR
    final_epoch: int                        =   0 # Last epoch to run
    lowest_error: float                     = 1e50
    lowest_error_epoch                      = 0
    backprop_headers                        = ["Config", "(*)", "Accp Blm", "=", "Raw Adj","LR", "=", "Final Adj"]
    input_scaling_used                      = "FixMe"
    #is_exploratory                          = False

    popup_headers                           = None #TODO Standardize these 4 names.
    popup_operators                         = None
    popup_finalizer_headers                 = None
    popup_finalizer_operators               = None

    def data_reset(self):
        seconds: float                          = 0.0
        cvg_condition: str                      = "Did Not Converge"
        learning_rate: float                    = 0.0
        final_epoch: int                        = 0                         # Last epoch to run
        lowest_error: float                     = 1e50
        lowest_error_epoch                      = 0
        backprop_headers                        = ["Config", "(*)", "Accp Blm", "=", "Raw Adj","LR", "=", "Final Adj"]

        popup_headers                           = None #TODO Standardize these 4 names.
        popup_operators                         = None
        popup_finalizer_headers                 = None
        popup_finalizer_operators               = None
        input_scaling_used                      = None

    def configure_popup_headers(self):
        (self.popup_headers, self.popup_operators,
         self.popup_finalizer_headers, self.popup_finalizer_operators) = self.optimizer.configure_optimizer(self)

    def set_defaults(self, training_data):
        #self.lego_selector = LegoSelector(self)
        self.training_data = training_data
        #print(f"Training data is_linear = {self.training_data.perceptron_ok}")
        self.smartNetworkSetup()
        #print(f"self.input_scaler - {self.input_scaler}")
        if self.input_scaler is not None:
            self.input_scaler.fit(training_data.get_inputs())
            print(f"scaled inputs = {self.input_scaler.scaled_data}")



    def smartNetworkSetup(self):
        self.lego_selector.apply(self, self.get_rules())        #print(f"pretty rules\n{self.lego_selector.pretty_print_applied_rules()}")

    def get_rules(self):        #  config.training_data.problem_type == " (0, 100, {"loss_function": Loss_BCEWithLogits}, "config.training_data.problem_type == 'Binary Decision'"),
        return [
            #(0, 100, {"architecture": [1]}, "training_data.perceptron_ok == 'True'"),
            (0, 100, {"output_activation": Activation_Sigmoid}, "training_data.problem_type == 'Binary Decision'"),
            (0, 100, {"output_activation": Activation_NoDamnFunction}, "training_data.problem_type != 'Binary Decision'"),
            (0, 200, {"loss_function": Loss_BCEWithLogits}, "output_activation.name == 'Sigmoid'"),
            #(0, 210, {"loss_function": Loss_MSE}, "training_data.has_high_outliers()"),
            #(0, 210, {"loss_function": Loss_MAE}, "not training_data.has_high_outliers()"),

            (0, 6691, {"optimizer": Optimizer_SGD}, "1 == 1"),
            (0, 6692, {"batch_mode": BatchMode.MINI_BATCH}, "1 == 1"),
            (0, 6693, {"batch_size": 1}, "1 == 1"),
            (0, 6694, {"architecture": [2, 1]}, "1 == 1"),
            (0, 6695, {"initializer": Initializer_Xavier}, "1 == 1"),
            (0, 6696, {"loss_function": Loss_MAE}, "1 == 1"),
            (0, 6697, {"hidden_activation": Activation_LeakyReLU}, "1 == 1"),
            (0, 6698, {"output_activation": Activation_NoDamnFunction}, "1 == 1"),
        ]


