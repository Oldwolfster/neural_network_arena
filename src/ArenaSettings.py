from dataclasses import dataclass
@dataclass
class HyperParameters:
    ############################################################
    # Battle Parameters are set here                           #
    ############################################################
    epochs_to_run           :int    = 500       # Number of times training run will cycle through all training data
    training_set_size       :int    = 500       # Qty of training data
    converge_epochs         :int    = 100       # How many epochs of no change before we call it converged?
    default_neuron_weight   :float  = .1        # Any initial value works as the training data will adjust it
    default_learning_rate   :float  = .0001     # Affects magnitude of weight adjustments
    converge_threshold      :float  = 0.01      # What percentage must weight be within compared to prior epochs weight to call it "same"
    accuracy_threshold      :float  = .2        # Careful - raising this to 1 or higher will break binary decision

    ############################################################
    # Report Parameters are set here                           #
    ############################################################
    display_train_data      :bool = True    # Display the training data at the end of the rn.
    display_graphs          :bool = False   # Display Graphs at the end of run
    display_epoch_sum       :bool = True    # Display the epoch summary
    display_logs            :bool = True    # Display the logs at the end of the run
    display_logs            :bool = False   # Display the logs at the end of the run

############################################################
# Choose Training Data Production Algorithm                #
############################################################
#training_pit              = "SingleInput_CreditScore"
#training_pit              = "CreditScoreRegression"
#training_pit              = "CreditScoreRegressionNeedsBias"
training_pit               = "Manual"
#training_pit              = "SalaryExperienceRegressionNeedsBias"

############################################################
# Choose Gladiators to Compete                             #
############################################################
gladiators = [
    '_Template_Simpletron_Regression'
    ,'Hayabusa2'
    #,'Regression_GBS'

]
