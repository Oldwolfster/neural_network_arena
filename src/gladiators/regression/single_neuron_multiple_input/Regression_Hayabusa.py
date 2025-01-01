from src.engine.Metrics import GladiatorOutput
from src.engine.BaseGladiator import Gladiator


class Hayabusa2_2inputs(Gladiator):
    """
    Hard coded to do 3 inputs the long way
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.bias = .5

    def training_iteration(self, training_data) -> GladiatorOutput:
        # Assign sample's inputs and target
        target = training_data[-1]
        inp0 = training_data[0]
        inp1 = training_data[1]
        #inp2 = training_data[2]

        # Step 1) Guess
        prediction      =  inp0 * self.weights[0]
        prediction      += inp1 * self.weights[1]
        #prediction     += inp2 * self.weights[2]
        prediction      += self.bias
        #
        error           = target - prediction               # Step 2) Check guess,
        adjustment      = error * self.learning_rate        # how far off * Learning rate

        self.weights[0] += inp0 * adjustment
        self.weights[1] += inp1 * adjustment
        #self.weights[2] += inp2 * adjustment
        self.bias       =  self.bias + adjustment

        return prediction


