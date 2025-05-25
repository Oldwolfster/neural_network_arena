from src.Legos.ActivationFunctions import *
from src.Legos.WeightInitializers import *
from src.Legos.Optimizers import *
from src.Legos.Scalers import *
from ..Legos.LossFunctions import *
from .RamDB import RamDB
from .TrainingData import TrainingData

from .convergence.ConvergenceDetector import ROI_Mode
from .. import NeuroForge
from ..ArenaSettings import HyperParameters
from ..Legos.LegoSelector import LegoSelector

from dataclasses import dataclass, field


class Config:
    # 🔹 Shared components for all models

    def __init__(self, hyper: HyperParameters, db: RamDB, training_data : TrainingData, gladiator_name:str):


        self.lego_selector: LegoSelector            = LegoSelector()
        self.training_data: TrainingData            = training_data # NOTE: training_data is stored in Config temporarily to support the rule engine.

        # 🔹 Unique components
        self.gladiator_name: str                    = gladiator_name
        self.optimizer                              = None #: Optimizer
        self.batch_mode: BatchMode                  = None
        self.batch_size: int                        = None
        self.architecture: list                     = None
        self.initializer: type                      = None
        self.loss_function: StrategyLossFunction    = None
        self.hidden_activation: type                = None  # Default to Tanh for regression and ReLU for BD
        self.output_activation: type                = None  # Will default to loss_function.recommended_output_activation if None
        self.target_scaler                          = None
        self.roi_mode                               = ROI_Mode.SWEET_SPOT
        self.bd_parameters                          = None # target a, target b, threshold.

        # Misc attributes  ##MAKE SURE TO ADD TO DATA RESET
        self.seconds: float                         = 0.0
        self.cvg_condition: str                     = "Did Not Converge"
        self.learning_rate: float                   = 0.0       # Read in beginning to instantiate  neurons with correct LR
        self.final_epoch: int                       = 0 # Last epoch to run
        self.lowest_error: float                    = 1e50
        self.lowest_error_epoch                     = 0
        self.exploratory                            = False
        self._percent_off                           = None
        self._accuracy_percent                      = None
        self.backprop_headers                       = ["Config", "(*)", "Accp Blm", "=", "Raw Adj","LR", "=", "Final Adj"]
        self.popup_headers                          = None #TODO Standardize these 4 names.
        self.popup_operators                        = None
        self.popup_finalizer_headers                = None
        self.popup_finalizer_operators              = None

        # Scaler properties
        self.scaler:        MultiScaler             = MultiScaler(training_data)
        self.target_scaler                          = None
        self.default_scalers                        = None
        self.default_input_scaler                   = None
        self._set_all_inputs                        = False  # backing field if needed

    @property
    def SetAllInputs(self): #TODO is this being used???
        return self._set_all_inputs

    @SetAllInputs.setter  #TODO is this being used???
    def SetAllInputs(self, value: bool):
        if value:
            # apply scaling logic across all inputs
            self.apply_default_scalers_to_all_inputs()
        self._set_all_inputs = value

    def configure_popup_headers(self):
        (self.popup_headers, self.popup_operators,
         self.popup_finalizer_headers, self.popup_finalizer_operators) = self.optimizer.configure_optimizer(self)

    def threshold_prediction(self, prediction): #not ideal home but until i find better...
        if not self.bd_parameters:  #Not BD so return same value.
            return prediction
        else:
            if prediction >= self.bd_parameters[2]: return self.bd_parameters[1]
            else: return  self.bd_parameters[0]

    def set_defaults(self, test_attribute = None, test_strategy = None):
        if test_attribute and test_strategy:
            setattr(self, test_attribute, test_strategy)
        self.smartNetworkSetup()
        if self.training_data.binary_decision and not self.bd_parameters:
            self.bd_parameters  = self.loss_function.bd_defaults

    def smartNetworkSetup(self):
        self.lego_selector.apply(self, self.get_rules(), self.exploratory)        #print(f"pretty rules\n{self.lego_selector.pretty_print_applied_rules()}")
        if self.default_scalers:
            self.scaler.set_all_input_scalers(Scaler_Robust)
        if self.target_scaler:
            self.scaler.set_target_scaler(self.target_scaler)


    def get_rules(self):        #  config.training_data.problem_type == " (0, 100, {"loss_function": Loss_BCEWithLogits}, "config.training_data.problem_type == 'Binary Decision'"),
        # Below fields...
        #   Allow overwrite, priority, field to set, value, condition to set it.
        return [
            #(0, 100, {"architecture": [1]}, "training_data.perceptron_ok == 'True'"),
            # First choose loss (based on problem type or custom override)
            (0, 200, {"loss_function"       : Loss_BCE}                     , "training_data.problem_type == 'Binary Decision'"),
            (0, 200, {"loss_function"       : Loss_MSE}                     , "training_data.problem_type != 'Binary Decision'"),
            (0, 300, {"output_activation"   : Activation_Sigmoid}           , "loss_function.name == 'Binary Cross-Entropy'"),
            (0, 300, {"output_activation"   : Activation_NoDamnFunction}    , "loss_function.name == 'Mean Squared Error'"),
            (0, 300, {"output_activation"   : Activation_Tanh}              , "loss_function.name == 'Hinge Loss'"),

            (0, 400, {"default_scalers"     : True}                         , "scaler.not_set_yet == False"),       #This one is different as ONE setting impacts scaler for ALL inputs
            (0, 500, {"target_scaler"       : Scaler_MinMax_Neg1to1}        , "output_activation.name == 'Tanh'"),
            (0, 500, {"target_scaler"       : Scaler_MinMax}                , "output_activation.name == 'Sigmoid'"),
            (0, 600, {"initializer"         : Initializer_He}               , "hidden_activation.name == 'LeakyReLU'"),

            #Below are default settings if an above rule has not set an option
            (0, 6691, {"optimizer"          : Optimizer_SGD}, "1 == 1"),
            (0, 6692, {"batch_mode"         : BatchMode.MINI_BATCH}, "1 == 1"),
            (0, 6693, {"batch_size"         : 1}, "1 == 1"),
            (0, 6694, {"architecture"       : [2, 1]}, "1 == 1"),

            (0, 6695, {"loss_function"      : Loss_MAE}, "1 == 1"),
            (0, 6696, {"hidden_activation"  : Activation_LeakyReLU}, "1 == 1"),
            (0, 6697, {"initializer"        : Initializer_Xavier}, "1 == 1"),
            (0, 6698, {"output_activation"  : Activation_NoDamnFunction}, "1 == 1"),
        ]
            #(0, 200, {"output_activation": Activation_Sigmoid}          , "training_data.problem_type == 'Binary Decision'"),
            #(0, 200, {"output_activation": Activation_NoDamnFunction}   , "training_data.problem_type != 'Binary Decision'"),
            #(0, 300, {"loss_function": Loss_BCE}                        , "output_activation.name == 'Sigmoid'"),
            #(0, 300, {"loss_function": Loss_MSE}                        , "output_activation.name == 'None'"),
            #(0, 210, {"loss_function": Loss_MSE}, "training_data.has_high_outliers()"),
            #(0, 210, {"loss_function": Loss_MAE}, "not training_data.has_high_outliers()"),

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

    @property
    def accuracy_percent(self):
        return 0
        if self._accuracy_percent is None:
            if self.training_data.problem_type == "Binary Decision":
                SQL_MAX_ACC = """
                    SELECT MAX(Accuracy) AS Accuracy
                    FROM EpochSummary
                    WHERE model_id = ?
                """
                result = self.db.query(SQL_MAX_ACC, (self.gladiator_name,))
                if result and result[0].get("Accuracy") is not None:
                    self._accuracy_percent = result[0]["Accuracy"]
                else:
                    self._accuracy_percent = 0.0
            else: # for regression it is percent within target
                if self.scaler.target_is_scaled:
                    mean_target = self.scaler.mean_target
                else:
                    mean_target = self.training_data.mean_absolute_target
                #ez_debug(mean_target=mean_target,lowest_err=self.lowest_error,isscaled=self.scaler.target_is_scaled)
                if mean_target == 0:
                    self._accuracy_percent = 0.0
                else:
                    raw_accuracy = 100.0 - ((self.lowest_error / mean_target) * 100)
                    self._accuracy_percent = max(0.0, min(100.0, raw_accuracy))
        return self._accuracy_percent