from dataclasses import dataclass
from src.NNA.Legos.ActivationFunctions import *
from src.NNA.Legos.WeightInitializers import *
from src.NNA.Legos.Optimizers import *
from src.NNA.Legos.LossFunctions import *


@dataclass
class HyperParameters:
    ############################################################
    # BATTLE Parameters are set here                           #
    ############################################################
    epochs_to_run           :int    = 11       # Number of times training run will cycle through all training data
    training_set_size       :int    = 30        # Qty of training data
    random_seed             :int    = 580636   # for seed 580636 - ONE EPOCH    #for seed 181026  DF LR 05 =9 but DF LR 4 = just 2 epochs    #for seed 946824, 366706 we got it in one!
    nf_count                :int    = 2
    display_train_data      :bool   = True        # Display the training data at the end of the rn.

dimensions={
    "loss_function"     : [Loss_MSE, Loss_MSE, Loss_HalfWit, Loss_BCE, Loss_BCE, Loss_Huber, Loss_Hinge,Loss_LogCosh],
    "hidden_activation" : "*",   # [Activation_Tanh, Activation_Sigmoid, Activation_LeakyReLU, Activation_ReLU, Activation_NoDamnFunction]
    "output_activation" : "*",   # [Activation_Tanh, Activation_Sigmoid, Activation_LeakyReLU, Activation_ReLU, Activation_NoDamnFunction]
    #"initializer"       : "*",
    #"architecture"      : [[4 , 4, 1], [2 , 2, 1]],
    #"optimizer"         : [Optimizer_SGD, Optimizer_Adam],  #suspect not working
    #"batch_size"        : [1,2,4,8,999] #I don't hink this one works yet
}

dimensions2={"loss_function": [Loss_MSE,Loss_BCE]}
dimensions2 = {"hidden_activation":[Activation_ReLU,Activation_LeakyReLU]}

dimensions={"architecture"      : [[4 , 4, 1],[2 , 2, 1]],}
dimensions={}#"loss_function"     : [Loss_MSE, Loss_Hinge, Loss_HalfWit, Loss_BCE]}
dimensions={}

############################################################
# ARENA -  Training Data Production Algorithm              #
############################################################
#"Regime_Trigger_Switch" #Impossible for normal FFNN
#,"Pathological_Discontinuous_Chaos"     #Not a dependent function
#,'Hidden_Switch_Power'        # Designed to fail... not a dependent function
arenas   = ['Iris_Two_Class','Predict_Income_2_Inputs']
arenas  = ['California_HousingUSD']
arenas2 = ['Nested_Sine_Flip']
arenas2  = ['Bit_Flip_Memory']
arenas = ['Titanic']

arenas  = [
#######################################################################
######################### Regression ##################################
#######################################################################
'Predict_Income_2_Inputs'
,'California_Housing'
,'California_HousingUSD'
,'One_Giant_Outlier'
,'One_Giant_OutlierExplainable'
,'Nested_Sine_Flip'
,'Chaotic_Function_Prediction'
,'Piecewise_Regime'
,'Adversarial_Noise'
,'MultiModal_Nonlinear_Interactions'
,'MultiModal_Temperature'

,'Delayed_Effect_BloodSugar'
,'Predict_EnergyOutput__From_Weather_Turbine'
,'Chaotic_Solar_Periodic'
,'Custom_Function_Recovery'
,'Deceptive_Multi_Regime_Entangler'
,'Predict_Income_2_Inputs_5Coefficents'
,'Predict_Income_2_Inputs_Nonlinear'
,'Predict_Income_Piecewise_Growth'
,'Customer_Churn_4X3'
,'AutoNormalize_Challenge'
,'Red_Herring_Features'
,'CarValueFromMiles'
,'Predict_MedicalCost_WithOutliers'
,'Target_Drift_Commodity'
,'Redundant_Features'

#######################################################################
######################### Binary Decision #############################
#######################################################################
,'Titanic'
,'SimpleBinaryDecision'
,'DefaultRisk__From_Income_Debt'
,'Bit_Flip_Memory'
,'Parity_Check'
,'XOR_Floats'
,'Sparse_Inputs'
,'Circle_In_Square'
,'XOR'
,'Moons'
,'Iris_Two_Class'
]
"""
"""

############################################################
# GLADIATORS - Neural Network Models to Compete            #
############################################################
gladiators = [ 'All_Defaults']
gladiators2 = [
    'All_Defaults'
    #,'BiggerIsNotBetter'
    #'Simplified_Descent_01_Solves_XOR_in_2'    #With 2 Layers. LR of 4 and seed  181026hits xor in 2 epochs.
    #'GBS'
    #'Simplex'
   #,'NeuroForge_Template'
    #'_Template_Simpletron'
]
