from typing import List
from dataclasses import dataclass, field
from src.Legos.ActivationFunctions  import *
from src.Legos.WeightInitializers   import *
from src.Legos.Optimizers           import *
from src.Legos.Scalers              import *
from src.Legos.LossFunctions        import *

# test = lister.list_legos("hidden_activation")
# test = lister.list_legos("output_activation")
# test = lister.list_legos("optimizer")
# test = lister.list_legos("scaler") # works a bit different....
# test = lister.list_legos("initializer")


@dataclass
class HyperParameters:
    ############################################################
    # BATTLE Parameters are set here                           #
    ############################################################
    record                  :bool   = True
    epochs_to_run           :int    = 100       # Number of times training run will cycle through all training data
    training_set_size       :int    = 30        # Qty of training data
    default_learning_rate   :float  = .1        # Affects magnitude of weight adjustments #.0001 Equalizer
    min_no_epochs           :int    = 0         # run for at least this many epochs
    display_train_data      :bool   = True        # Display the training data at the end of the rn.
    run_neuroForge          :bool   = True
    random_seed             :int    = 599059 #748141
    #160995   #181026 #393828 #874170  331670
    # for seed 580636 - ONE EPOCH
    #for seed 181026  DF LR 05 =9 but DF LR 4 = just 2 epochs     #241709 LR1 converges in 24 friggen epochs!
    #for seed 946824, 366706 we got it in one!
    #311161 gets 3 epochs with adaptive LR and explosion threshold of 5.
    #375655 get 2 epochs

    ##############################################################
    # Convergence Thresholds                                     #
    ##############################################################
    threshold_Signal_MostAccurate   = .001
    threshold_Signal_Economic       = .3        # The larger the quicker it converges
    threshold_Signal_SweetSpot      = .01       # The larger the quicker it converges
    converge_epochs         :int    = 10       # How many epochs of no change before we call it converged?
    converge_threshold      :float  = 1e-12      # What percentage must MAE be within compared to prior epochs MAE to call it "same" #.001 Equalizer
    accuracy_threshold      :float  = .000005        # In regression, how close must it be to be considered "accurate" - Careful - raising this to 1 or higher will break binary decision
    data_labels                     = []        # List to hold the optional data labels

    ##############################################################
    # NEW REPORT Parameters are set here - GOING TO MULT NEURONS #
    ##############################################################
    criteria_neuron :List[int] = None       # None = show all - otherwise list of neurons to include
    criteria_weight :List[int] = None       # None = show all - 0= Ommit weights - Otherwise list all weights to include


dimensions={
    #"loss_function": [Loss_MSE, Loss_BCE, Loss_MAE],
    # "hidden_activation": [Activation_ReLU, Activation_Tanh],
    # Add more here as needed!
}

############################################################
# GLADIATORS - Neural Network Models to Compete            #
############################################################
gladiators = [
    #'All_Defaults'
    'BiggerIsNotBetter'
    #'Simplified_Descent_01_Solves_XOR_in_2'    #With 2 Layers. LR of 4 and seed  181026hits xor in 2 epochs.
    #'GBS'
    #'Simplex'
    #'NeuroShepherd'
    #,'Simplex2'
    #'GBS'
    #'BCEWL_LR_Sweep'
   # 'BCEWL_LR_Set_to_point1'
    #'Hand_Tuned'
    #,'test_newOpt'
   #,'NeuroForge_Template'


    #,'Test_BCE'
    #'Test_BCE'
    #,'TestBatch'
    #,'Adam_Template'
   # 'Adam_XOR_DEBUG_2'
    #'NeuroForge_CaliforniaHousing'
    #'GBS_Baseline'
    #'GBS_XOR'
    #'HayabusaNormalizer'
    #'_Template_Simpletron'
]

############################################################
# ARENA -  Training Data Production Algorithm              #
############################################################
training_pit                = "XOR"
#training_pit                = "CaliforniaHousePricesUSD"
training_pit                = "Predict_Income_2_Inputs"
#training_pit="Predict_Income_2_Inputs_Nonlinear"
#training_pit="Predict_EnergyOutput__From_Weather_Turbine"
#training_pit="Predict_TrafficFlow__From_Weather_Time_Events"
#training_pit="StockPrice__From_Indicators"
#training_pit                = "SimpleBinaryDecision"  # Single Input Credit score
#training_pit              = "Arena_CenteredData"
#training_pit                = "Predict_MedicalCost_WithOutliers"
#training_pit               = "Titanic"
#training_pit                = "Predict_Income_Piecewise_Growth"
#training_pit                = "Moons"
#training_pit                = "Manual"
#training_pit                = "Predict_Income_2_Inputs_5Coefficents"
##training_pit              = "Salary2InputsNonlinear"
#training_pit                = "California_Housing"
#training_pit                = "Customer_Churn_4X3"
#training_pit = "CarValueFromMiles"
training_pit = "DefaultRisk__From_Income_Debt"
training_pit = "DefaultRisk__From_Income_Debt"
######## Tests
#"Regime_Trigger_Switch" #Impossible for normal FFNN
#,"Pathological_Discontinuous_Chaos"     #Not a dependent function
#,'Hidden_Switch_Power'        # Designed to fail... not a dependent function
#batch_arenas         = [
arenas         = [
"DefaultRisk__From_Income_Debt"

 ]
"""   
,"Deceptive_Multi_Regime_Entangler"
,"MultiModal_Nonlinear_Interactions"
,"AutoNormalize_Challenge"
,"Titanic"

,"Bit_Flip_Memory"
,"Iris_Two_Class"
,"Chaotic_Function_Prediction"
,"Circle_In_Square"
,"Custom_Function_Recovery"
,"One_Giant_Outlier"
,"Parity_Check"
,"Sparse_Inputs"
, 'Adversarial_Noise'       #BiggerIsNotBetter worked great, all defaults gradient exploded (with 4,4 architecture
,'Chaotic_Solar_Periodic'    #89
, 'Delayed_Effect_BloodSugar'  #4,4, with target scaling -TS hurt
,'MultiModal_Temperature'
,'Piecewise_Regime'
,'Red_Herring_Features'
,'Redundant_Features'
,'Target_Drift_Commodity'
,'XOR_Floats'
,'Nested_Sine_Flip'
,'DefaultRisk__From_Income_Debt'    

"""



############################################################
# List or use training data from a previous run            #
############################################################
instead_of_run_show_past_runs       = False                 # Get list of prior runs
#instead_of_run_show_past_runs      = True                   # Get list of prior runs
run_previous_training_data          = ''                    # Use training data from past run
#run_previous_training_data          = '20241028_195936'    # Use training data from past run

#21.371165216310544
#35.41909166991863
