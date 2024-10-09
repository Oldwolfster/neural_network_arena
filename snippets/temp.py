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
