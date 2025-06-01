from typing import List
from dataclasses import dataclass, field

@dataclass
class HyperParameters:
    ############################################################
    # BATTLE Parameters are set here                           #
    ############################################################
    epochs_to_run           :int    = 100       # Number of times training run will cycle through all training data
    training_set_size       :int    = 30        # Qty of training data
    default_learning_rate   :float  = .1        # Affects magnitude of weight adjustments #.0001 Equalizer
    nf_count                :int    = 2
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
    #"loss_function": [Loss_MSE,Loss_BCE],
    #"loss_function": "*" ,
    #"hidden_activation": [Activation_ReLU, Activation_Tanh],
    #"initializer": "*"
    #"output_activation": *
    #"batch_size":[1,2,4,8,999]
    #"architecture":[[4 , 4, 1], [2 , 2, 1]]
}
#dimensions={}

############################################################
# GLADIATORS - Neural Network Models to Compete            #
############################################################
gladiators = [
    'All_Defaults'

    #,'BiggerIsNotBetter'
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
    #'_Template_Simpletron'
]

############################################################
# ARENA -  Training Data Production Algorithm              #
############################################################
#"Regime_Trigger_Switch" #Impossible for normal FFNN
#,"Pathological_Discontinuous_Chaos"     #Not a dependent function
#,'Hidden_Switch_Power'        # Designed to fail... not a dependent function
arenas   = ['Iris_Two_Class','Predict_Income_2_Inputs']
arenas2   = ['Iris_Two_Class']
arenas2   = [
"Titanic"

,'Predict_Income_2_Inputs'
,'MultiModal_Temperature'
,"Parity_Check"
,"Iris_Two_Class"


,'Deceptive_Multi_Regime_Entangler'
,'DefaultRisk__From_Income_Debt'
,'Moons'
,'CarValueFromMiles'
,"California_Housing"
,"SimpleBinaryDecision"
,"Customer_Churn_4X3"
,"Predict_EnergyOutput__From_Weather_Turbine"
,"Predict_Income_2_Inputs_Nonlinear"

,'Predict_Income_Piecewise_Growth'
,'Predict_Income_2_Inputs_5Coefficents'
,"MultiModal_Nonlinear_Interactions"
,"AutoNormalize_Challenge"

,"CaliforniaHousePricesUSD"
,"Bit_Flip_Memory"

,'XOR'
,"Chaotic_Function_Prediction"
,"Circle_In_Square"
,"Custom_Function_Recovery"
,"One_Giant_Outlier"

,"Sparse_Inputs"
, 'Adversarial_Noise'       #BiggerIsNotBetter worked great, all defaults gradient exploded (with 4,4 architecture
,'Chaotic_Solar_Periodic'    #89
, 'Delayed_Effect_BloodSugar'  #4,4, with target scaling -TS hurt

,'Piecewise_Regime'
,'Red_Herring_Features'
,'Redundant_Features'
,'Target_Drift_Commodity'
,'XOR_Floats'
,'Nested_Sine_Flip'
,'DefaultRisk__From_Income_Debt'
,'Predict_MedicalCost_WithOutliers'
  ]
"""
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
