
from typing import List


from dataclasses import dataclass, field




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

    epochs_to_run           :int    = 1222      # Number of times training run will cycle through all training data
    training_set_size       :int    = 2      # Qty of training data
    default_learning_rate   :float  = .5       # Affects magnitude of weight adjustments #.0001 Equalizer
    min_no_epochs           :int    = 0        # run for at least this many epochs
    gradient_clip_threshold :float  = 1e12
    #seed that is all green 529966
    #241709 LR1 converges in 24 friggen epochs!

    random_seed             :int       = 723182 #393828 #874170
    #507529 hinge with .1 lr in 10 epochs.
#171193 #8422    # Put zero to NOT use seed!  12345 and LR 1 and 5 or 2,3 arch giving me overflow
    #965512 causes error #621575 LR1 gets stuck and free, but itlooks like it doesn't converge properly
    converge_epochs         :int    = 10       # How many epochs of no change before we call it converged?
    converge_threshold      :float  = .001      # What percentage must MAE be within compared to prior epochs MAE to call it "same" #.001 Equalizer
    accuracy_threshold      :float  = .1        # In regression, how close must it be to be considered "accurate" - Careful - raising this to 1 or higher will break binary decision
    data_labels                     = []        # List to hold the optional data labels

    ############################################################
    # REPORT Parameters are set here                           #
    ############################################################
    color_neurons           :bool = True    # Display the training data at the end of the rn.
    display_train_data      :bool = True    # Display the training data at the end of the rn.
    display_graphs          :bool = True   # Display Graphs at the end of run
    display_graphs          :bool = False   # Display Graphs at the end of run
    display_epoch_sum       :bool = True    # Display the epoch summary
    display_neuron_report   :bool = False   # Display the logs at the end of the run
    display_logs            :bool = True   # Display the logs at the end of the run
    detail_log_min          :int  = 0       # Which epochs to display detailed logs for(min)
    detail_log_max          :int  = 10000       # Which epochs to display detailed logs for(min)
    run_neuroForge          :bool = True
    ##############################################################
    # NEW REPORT Parameters are set here - GOING TO MULT NEURONS #
    ##############################################################
    criteria_neuron :List[int] = None       # None = show all - otherwise list of neurons to include
    criteria_weight :List[int] = None       # None = show all - 0= Ommit weights - Otherwise list all weights to include

############################################################
# ARENA -  Training Data Production Algorithm              #
############################################################

#training_pit              = "Predict_Income_2_Inputs"

training_pit                = "XOR"
#training_pit                = "SimpleBinaryDecision"  # Single Input Credit score
#training_pit              = "SimpleBinaryDecision"
#training_pit              = "Predict_Income_2_Inputs"
##training_pit              = "Salary2InputsNonlinear"
#training_pit              = "Manual"

############################################################
# GLADIATORS - Neural Network Models to Compete            #
############################################################
gladiators = [
    #'Blackbird'
    #'XOR_Hayabusa'
    #'XOR_Linear'
    #'NeuroForge_Template'
    'NeuroForge_Template_BCE'
    #'NeuroForge_4Layers'
    #,'NeuroForge_Experiment'
   #'_Template_Simpletron'
    #,'HayabusaTwoWeights'   #2 weights
    #'XORTutorial'
    #'GBS_XOR'
    #'HayabusaNormalizer'
    #'HayabusaTwoNeurons'
    #'HayabusaDrawTwoNeurons'
    #'HayabusaDrawMultipleLayers'
    #'GBS_MultipleOutputNeurons'
    #'GBS_one_neuron'
    #'_Template_Simpletron'

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
