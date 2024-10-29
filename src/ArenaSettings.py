from dataclasses import dataclass
@dataclass
class HyperParameters:
    ############################################################
    # BATTLE Parameters are set here                           #
    ############################################################
    epochs_to_run           :int    = 500       # Number of times training run will cycle through all training data
    training_set_size       :int    = 100       # Qty of training data
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
training_pit              = "CreditScoreRegressionNeedsBias"
training_pit               = "Manual"
#training_pit              = "SalaryExperienceRegressionNeedsBias"
training_pit              = "Salary2Inputs"
#training_pit              = "Salary2Inputs_B"
#training_pit              = "Salary2Inputs_C"


############################################################
# GLADIATORS - Neural Network Models to Compete            #
############################################################
gladiators = [
    '_Template_Simpletron_Regression_SNMI'
    #,'Hayabusa2'
    #,'_Template_Simpletron_Regression2inputs'
    #,'Hayabusa2_2inputs'
    ,'Regression_GBS_SNMI'
    #,'Regression_GBS_MultInputs_B'
    #,'Regression_GBS_2_Inputs'
    #,'Regression_GBS_2_InputsB'
]

############################################################
# List or use training data from a previous run            #
############################################################
instead_of_run_show_past_runs       = False                 # Get list of prior runs
#instead_of_run_show_past_runs      = True                   # Get list of prior runs
run_previous_training_data          = ''                    # Use training data from past run
#run_previous_training_data          = '20241028_195936'    # Use training data from past run

#21.371165216310544
#35.41909166991863
