from src.Legos.ActivationFunctions import *
from .RamDB import RamDB
from .TrainingData import TrainingData
from src.Legos.WeightInitializers import *
from src.Legos.Optimizers import *
from src.Legos.Scalers import *
from .convergence.ConvergenceDetector import ROI_Mode
from ..ArenaSettings import HyperParameters
from ..Legos.LegoSelector import LegoSelector
from ..Legos.LossFunctions import *
from dataclasses import dataclass, field


class Config:
    # 🔹 Shared components for all models

    def __init__(self, hyper: HyperParameters, db: RamDB, training_data : TrainingData, gladiator_name:str):

        self.hyper:         HyperParameters         = hyper
        self.lego_selector: LegoSelector            = LegoSelector()
        self.db:            RamDB                   = db
        self.training_data: TrainingData            = training_data
        self.scaler:        MultiScaler             = MultiScaler(training_data)


        # 🔹 Unique components
        self.gladiator_name: str                     = gladiator_name
        self.optimizer                               = None #: Optimizer
        self.batch_mode: BatchMode                   = None
        self.batch_size: int                         = None
        self.architecture: list                      = None
        self.initializer: type                       = None
        self.loss_function: LossFunction             = None
        self.hidden_activation: type                 = None  # Default to Tanh for regression and ReLU for BD
        self.output_activation: type                 = None  # Will default to loss_function.recommended_output_activation if None
        self.target_scaler                           = Scaler_NONE
        self.roi_mode                                = ROI_Mode.SWEET_SPOT
        self.bd_parameters                           = None # target a, target b, threshold.

        # Misc attributes  ##MAKE SURE TO ADD TO DATA RESET
        self.seconds: float                          = 0.0
        self.cvg_condition: str                      = "Did Not Converge"
        self.learning_rate: float                       = 0.0       # Read in beginning to instantiate  neurons with correct LR
        self.final_epoch: int                        =   0 # Last epoch to run
        self.lowest_error: float                     = 1e50
        self.lowest_error_epoch                      = 0
        self._percent_off                               = None
        self.default_scalers                        = None
        self.backprop_headers                        = ["Config", "(*)", "Accp Blm", "=", "Raw Adj","LR", "=", "Final Adj"]
        #is_exploratory                          = False
        self.popup_headers                           = None #TODO Standardize these 4 names.
        self.popup_operators                         = None
        self.popup_finalizer_headers                 = None
        self.popup_finalizer_operators               = None
        self.default_scalers                        = None

    def data_reset_Deleteme(self):
        self.seconds: float                          = 0.0
        self.cvg_condition: str                      = "Did Not Converge"
        self.learning_rate: float                    = 0.0
        self.final_epoch: int                        = 0                         # Last epoch to run
        self.lowest_error: float                     = 1e50
        self.lowest_error_epoch                      = 0
        self.backprop_headers                        = ["Config", "(*)", "Accp Blm", "=", "Raw Adj","LR", "=", "Final Adj"]

        self.popup_headers                           = None #TODO Standardize these 4 names.
        self.popup_operators                         = None
        self.popup_finalizer_headers                 = None
        self.popup_finalizer_operators               = None
    @property
    def neuroforge_layers(self):
        """
        Returns a structured list of layer definitions for visualization.
        Each item is a list of LayerComponent objects: either a Decider, Scaler, or Threshold.
        For right now, we will just use a str
        """
        layers = []

        # Input scaler(s)
        if self.input_scaler != Scaler_NONE:
            layers.append("InputScalar")
            #layers.append([ScalerComponent(name=self.input_scaler.name) for _ in range(self.input_count)])

        # Main architecture
        for n in self.architecture_core:
            layers.append("Decider")
            #layers.append([DeciderComponent() for _ in range(n)])


        # Threshold neuron (optional, e.g. for binary decisions)
        #if self.uses_threshold:
        #    layers.append([ThresholdComponent()])

        return layers


    @property
    def percent_off(self):
        if self._percent_off is None:
            if self.training_data.problem_type == "Binary Decision":
                SQL_MAX_OFF = """
                    SELECT MAX(Accuracy) AS Accuracy
                    FROM EpochSummary
                    WHERE model_id = ?
                """
                result = self.db.query(SQL_MAX_OFF, (self.gladiator_name,))
                if result and result[0].get("Accuracy") is not None:
                    self._percent_off = 100.0 - (result[0]["Accuracy"] )
                else:
                    self._percent_off = 100.0
            else:
                mean_target = self.training_data.mean_absolute_target
                if mean_target == 0:
                    self._percent_off = 100.0
                else:
                    self._percent_off = (self.lowest_error / mean_target) * 100
        return self._percent_off

    def configure_popup_headers(self):
        (self.popup_headers, self.popup_operators,
         self.popup_finalizer_headers, self.popup_finalizer_operators) = self.optimizer.configure_optimizer(self)

    def threshold_prediction(self, prediction): #not ideal home but until i find better...
        if not self.bd_parameters:  #Not BD so return same value.
            return prediction
        else:
            if prediction >= self.bd_parameters[2]: return self.bd_parameters[1]
            else: return  self.bd_parameters[0]


    def set_defaults(self):
        self.smartNetworkSetup()
        if self.training_data.binary_decision and not self.bd_parameters:
            self.bd_parameters  = self.loss_function.bd_defaults

    def smartNetworkSetup(self):
        self.lego_selector.apply(self, self.get_rules())        #print(f"pretty rules\n{self.lego_selector.pretty_print_applied_rules()}")
        if self.default_scalers:
            self.scaler.set_all_input_scalers(Scaler_Robust)
        #print(f"Defaults applied:  Architecture ->{self.architecture}")

    def get_rules(self):        #  config.training_data.problem_type == " (0, 100, {"loss_function": Loss_BCEWithLogits}, "config.training_data.problem_type == 'Binary Decision'"),
        return [
            #(0, 100, {"architecture": [1]}, "training_data.perceptron_ok == 'True'"),
            (0, 100, {"output_activation": Activation_Sigmoid}, "training_data.problem_type == 'Binary Decision'"),
            (0, 100, {"output_activation": Activation_NoDamnFunction}, "training_data.problem_type != 'Binary Decision'"),
            (0, 200, {"loss_function": Loss_BCEWithLogits}, "output_activation.name == 'Sigmoid'"),
            (0, 200, {"loss_function": Loss_MSE}, "output_activation.name == 'None'"),
            (0, 300, {"default_scalers": True}, "scaler.not_set_yet == False"),
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


