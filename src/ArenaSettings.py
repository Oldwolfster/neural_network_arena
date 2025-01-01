from dataclasses import dataclass
from typing import List


@dataclass
class ReportSelection:
    ############################################################
    # Report Parameters are set here                           #
    ############################################################
    Training_Evolution_with_Multiple_Scales :bool   = True

@dataclass
class HyperParameters:
    ############################################################
    # BATTLE Parameters are set here                           #
    ############################################################
    epochs_to_run           :int    = 5555         # Number of times training run will cycle through all training data
    training_set_size       :int    = 111       # Qty of training data
    converge_epochs         :int    = 2       # How many epochs of no change before we call it converged?
    default_neuron_weight   :float  = .0        # Any initial value works as the training data will adjust it
    default_learning_rate   :float  = .0001     # Affects magnitude of weight adjustments #.0001 Equalizer
    # .1 get's 4k    .1446456=4k      .14464565=29  .14= 29
    #                                 .14464564
    converge_threshold      :float  = .001      # What percentage must MAE be within compared to prior epochs MAE to call it "same" #.001 Equalizer
    accuracy_threshold      :float  = .1        # In regression, how close must it be to be considered "accurate" - Careful - raising this to 1 or higher will break binary decision

    ############################################################
    # REPORT Parameters are set here                           #
    ############################################################
    display_train_data      :bool = True    # Display the training data at the end of the rn.
    display_graphs          :bool = True   # Display Graphs at the end of run
    display_graphs          :bool = False   # Display Graphs at the end of run
    display_epoch_sum       :bool = True    # Display the epoch summary
    display_logs            :bool = False   # Display the logs at the end of the run
    #display_logs            :bool = True   # Display the logs at the end of the run
    detail_log_min          :int  = 0       # Which epochs to display detailed logs for(min)
    detail_log_max          :int  = 10000       # Which epochs to display detailed logs for(min)

    ##############################################################
    # NEW REPORT Parameters are set here - GOING TO MULT NEURONS #
    ##############################################################
    criteria_neuron :List[int] = None       # None = show all - otherwise list of neurons to include
    criteria_weight :List[int] = None       # None = show all - 0= Ommit weights - Otherwise list all weights to include



############################################################
# ARENA -  Training Data Production Algorithm              #
############################################################

training_pit              = "Predict_Income_2_Inputs"
training_pit              = "Manual"
#training_pit              = "CreditScoreRegression"
#training_pit              = "Predict_Income_2_Inputs"
##training_pit              = "Salary2InputsNonlinear"




############################################################
# GLADIATORS - Neural Network Models to Compete            #
############################################################
gladiators = [
    #"HayabusaFixed"
    #,'Blackbird'
    #'Hayabusa'
    #'HayabusaNormalizer'
    'HayabusaTwoNeurons'
    ,'GBS_MultipleOutputNeurons'
    #'GBS_one_neuron'

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
