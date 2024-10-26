from dataclasses import dataclass
@dataclass
class HyperParameters:
    ############################################################
    # BATTLE Parameters are set here                           #
    ############################################################
    epochs_to_run           :int    = 500       # Number of times training run will cycle through all training data
    training_set_size       :int    = 10       # Qty of training data
    converge_epochs         :int    = 100       # How many epochs of no change before we call it converged?
    default_neuron_weight   :float  = .1        # Any initial value works as the training data will adjust it
    default_learning_rate   :float  = .001     # Affects magnitude of weight adjustments
    converge_threshold      :float  = 0.01      # What percentage must weight be within compared to prior epochs weight to call it "same"
    accuracy_threshold      :float  = .2        # Careful - raising this to 1 or higher will break binary decision

    ############################################################
    # REPORT Parameters are set here                           #
    ############################################################
    display_train_data      :bool = True   # Display the training data at the end of the rn.
    display_graphs          :bool = False   # Display Graphs at the end of run
    display_epoch_sum       :bool = True    # Display the epoch summary
    display_logs            :bool = True    # Display the logs at the end of the run
    #display_logs            :bool = False   # Display the logs at the end of the run

############################################################
# ARENA -  Training Data Production Algorithm              #
############################################################
training_pit              = "SingleInput_CreditScore"
#training_pit              = "CreditScoreRegression"
#training_pit              = "CreditScoreRegressionNeedsBias"
training_pit               = "Manual"
#training_pit              = "SalaryExperienceRegressionNeedsBias"
#training_pit              = "Salary2Inputs"
#training_pit              = "Salary2Inputs_B"
#training_pit              = "Salary2Inputs_C"


############################################################
# GLADIATORS - Neural Network Models to Compete            #
############################################################
gladiators = [
    #'_Template_Simpletron_Regression'
    #,'Hayabusa2'
    #,'_Template_Simpletron_Regression2inputs'
    #,'Hayabusa2_2inputs'
    'Regression_GBS'
    ,'Regression_GBS_2_Inputs'
    #,'Regression_GBS_2_InputsB'

]
