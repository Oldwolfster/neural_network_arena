"""
Good morning, i'd like to make big progress on the NNA today.  It was a fun and educational diversion building the visualizations that animates and illustrates the training process through neuron coloring and completing backprop for SGD.

        # Additional options
        Optimizers (Adam, RMSProp, what else?)
        Regularization / Dropout Rate
        Momentum
        Convergence
        Metrics
        Verbosity
        Learning Rate Schedule
        Batch Size
        Random_Seed
        Gradiant Clipping parameters
        #what other options should be here?

Documentation:
Instead of commenting out methods, I suggest:

Create a comprehensive example model with inline documentation
Add Sphinx-style docstrings
Create a model "cookbook" showing common patterns
Add validation for configuration option

class ModelConfig:
    def __init__(self,
                 # Architecture
                 layer_sizes: List[int],
                 activation_functions: Union[Callable, List[Callable]],
                 weight_initializers: Union[Callable, List[Callable]],

                 # Training
                 learning_rate: float = 0.01,
                 batch_size: int = 32,
                 max_epochs: int = 1000,

                 # Optimization
                 optimizer: str = 'sgd',  # ['sgd', 'adam', 'rmsprop']
                 momentum: float = 0.0,
                 beta1: float = 0.9,  # Adam param
                 beta2: float = 0.999, # Adam param
                 epsilon: float = 1e-8, # Optimizer stability

                 # Regularization
                 l1_lambda: float = 0.0,
                 l2_lambda: float = 0.0,
                 dropout_rate: float = 0.0,

                 # Learning Rate Scheduling
                 lr_schedule: Optional[Callable] = None,
                 lr_patience: int = 10,
                 lr_factor: float = 0.5,
                 min_lr: float = 1e-6,

                 # Early Stopping
                 early_stopping_patience: int = 20,
                 min_delta: float = 1e-4,

                 # Gradient Clipping
                 clip_norm: Optional[float] = None,
                 clip_value: Optional[float] = None,

                 # Misc
                 random_seed: Optional[int] = None,
                 verbose: int = 1,  # 0=silent, 1=progress bar, 2=one line per epoch

                 # Callbacks
                 callbacks: List[Callable] = None,

                 # Metrics
                 metrics: List[str] = None  # ['accuracy', 'precision', 'recall', etc]
                ):
        # Validation and initialization code here


**********************************************************************************************************************************
**********************************************************************************************************************************
**********************************************************************************************************************************
**********************************************************************************************************************************
from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class ModelConfig:
    learning_rate: float = 0.01
    batch_size: int = 32
    epochs: int = 100
    _database_connection: Optional[Any] = None  # Private field

    @property
    def database_connection(self) -> Optional[Any]:
        return self._database_connection

    # No setter for database_connection, making it read-only

# Example usage
config = ModelConfig(_database_connection="my_db_connection")

# This works
print(config.database_connection)  # Output: my_db_connection

# This will raise an error because database_connection is read-only
config.database_connection = "new_db_connection"  # Error: can't set attribute


#################The superclass (BaseGladiator) still accepts a ModelConfig object:
class BaseGladiator:
    def __init__(self, config: ModelConfig):
        # Assign configuration values to instance attributes
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.regularization = config.regularization
        self.momentum = config.momentum
        self.early_stopping = config.early_stopping
        self.metrics = config.metrics
        self.verbose = config.verbose
        self.training_data = config.training_data
        self.validation_data = config.validation_data
        self.database_connection = config.database_connection

        # Initialize neurons with default settings
        self.initialize_neurons([4, 4, 5], [Initializer_Xavier], Tanh)
        Neuron.layers(1).set_activation(ReLU)
        self.neurons(1, 1).set_activation(Tanh)

#############################The framework creates and passes the ModelConfig object:
# Framework-level configuration
framework_config = ModelConfig(
    learning_rate=0.01,
    batch_size=32,
    epochs=100,
    regularization='L2',
    momentum=0.9,
    early_stopping={'patience': 5},
    metrics=['accuracy', 'loss'],
    verbose=True,
    training_data=load_training_data(),
    validation_data=load_validation_data(),
    database_connection=create_database_connection(),
)
*********************Simple child classes
class ChildClass(BaseGladiator):
    def __init__(self, config: ModelConfig):
        # Override specific parameters if needed
        config.learning_rate = 0.001  # Custom learning rate
        config.batch_size = 64  # Custom batch size

        # Pass the configuration to the superclass
        super().__init__(config)

"""