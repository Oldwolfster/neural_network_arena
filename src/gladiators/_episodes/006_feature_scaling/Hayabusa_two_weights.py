from src.gladiators.BaseGladiator import Gladiator


class SuzukiHayabusaTwoWeights(Gladiator):
    """
    A simple single input regression model
    This version will utilize both weights
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.initialize_neurons(2)
        #self.training_data.set_normalization_min_max()

    def training_iteration(self, training_data) -> float:
        inp_0 = training_data[0]  # First input
        inp_1 = training_data[1]  # Second input
        target = training_data[-1]  # Target value

        # Step 1: Compute the output of each neuron
        output_0 = (inp_0 * self.neurons[0].weights[0] +
                    inp_1 * self.neurons[0].weights[1] +
                    self.neurons[0].bias)

        output_1 = (inp_0 * self.neurons[1].weights[0] +
                    inp_1 * self.neurons[1].weights[1] +
                    self.neurons[1].bias)

        # Combine the outputs to make the final prediction
        prediction = (output_0 + output_1) / 2  # Example: averaging their outputs

        # Step 2: Calculate the error
        error = target - prediction

        # Step 3: Update weights and biases for both neurons
        # Neuron 0 updates
        self.neurons[0].weights[0] += error * self.learning_rate * inp_0
        self.neurons[0].weights[1] += error * self.learning_rate * inp_1
        self.neurons[0].bias += error * self.learning_rate

        # Neuron 1 updates
        self.neurons[1].weights[0] += error * self.learning_rate * inp_0
        self.neurons[1].weights[1] += error * self.learning_rate * inp_1
        self.neurons[1].bias += error * self.learning_rate

        return prediction
