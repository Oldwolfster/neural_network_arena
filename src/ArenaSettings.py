############################################################
# Battle Parameters are set here                           #
############################################################
epochs_to_run           = 500       # Number of times training run will cycle through all training data
training_set_size       = 500      # Qty of training data
default_neuron_weight   = .1        # Any initial value works as the training data will adjust it
converge_epochs         = 100       # How many epochs of no change before we call it converged?
default_learning_rate   = .0001     # Affects magnitude of weight adjustments
converge_threshold      = 0.01      # What percentage must weight be within compared to prior epochs weight to call it "same"
accuracy_threshold      = .2        # Careful - raising this to 1 or higher will break binary decision

############################################################
# Report Parameters are set here                           #
############################################################
display_train_data  = True         # Display the training data at the end of the rn.
display_graphs      = False         # Display Graphs at the end of run
display_epoch_sum   = True          # Display the epoch summary
display_logs        = True          # Display the logs at the end of the run
display_logs        = False         # Display the logs at the end of the run

############################################################
# Choose Training Data Production Algorithm                #
############################################################
#training_pit = "SingleInput_CreditScore"
#training_pit = "CreditScoreRegression"
#training_pit = "CreditScoreRegressionNeedsBias"
training_pit = "Manual"
#training_pit = "SalaryExperienceRegressionNeedsBias"

############################################################
# Choose Gladiators to Compete                             #
############################################################
gladiators = [
    '_Template_Simpletron_Regression'
    ,'Hayabusa2'
    #,'Regression_GBS'

]
