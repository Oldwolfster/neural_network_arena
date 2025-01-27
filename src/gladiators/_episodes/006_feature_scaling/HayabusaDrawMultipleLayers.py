from src.engine.BaseGladiator import Gladiator


class HayabusaDrawMultipleLayers(Gladiator):
    """
    This upgrade supports two neurons
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.initialize_neurons([3,4,5])
        self.normalizers = self.training_data.normalizers  # Output: [0.333, 0.666]
        #self.training_data.set_normalization_min_max()

    def training_iteration(self, training_data) -> float:
        inp_0 = training_data[0]  # First input
        inp_1 = training_data[1]  # Second input
        target = training_data[-1]  # Target value

        # Step 1: Compute the output of the first neuron
        output_0 = (inp_0 * self.neurons[0].weights[0] +
                    inp_1 * self.neurons[0].weights[1] +
                    self.neurons[0].bias)

        #print(f"Initialized Neurons: {len(self.neurons)}")
        #for i, neuron in enumerate(self.neurons):
        #    print(f"Neuron {i}: Weights = {neuron.weights}, Bias = {neuron.bias}")

        # Step 2: Compute the output of the second neuron
        output_1 = (inp_0 * self.neurons[1].weights[0] +
                    inp_1 * self.neurons[1].weights[1] +
                    self.neurons[1].bias)

        # sum, should we divide???? Dividing adds epochs for same result... strange  in Predict_Income_2_Inputs
        prediction = (output_0 + output_1)

        # Step 2: Calculate the error
        error = target - prediction

        #print(f"in hayabusatwoneurontron Before: self.neurons[0].weights[0]={self.neurons[0].weights[0]}\tself.neurons[0].weights[1]={self.neurons[0].weights[1]}" )
        # Step 3: Update weights and biases for the first neuron
        self.neurons[0].weights[0] += error * self.learning_rate * self.normalizers[0]
        self.neurons[0].weights[1] += error * self.learning_rate * self.normalizers[1]
        self.neurons[0].bias += error * self.learning_rate

        # Step 4: Update weights and biases for the first neuron
        self.neurons[1].weights[0] += error * self.learning_rate * self.normalizers[0]
        self.neurons[1].weights[1] += error * self.learning_rate * self.normalizers[1]
        self.neurons[1].bias += error * self.learning_rate
        #print(f"in hayabusatwoneurontron AFTER: self.neurons[0].weights[0]={self.neurons[0].weights[0]}\tself.neurons[0].weights[1]={self.neurons[0].weights[1]}" )
        # (Second neuron deliberately ignored)

        return prediction


    def training_iterationTwoNeuronBEATSGBS(self, training_data) -> float:
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
        prediction = output_0
        # Step 2: Calculate the error
        error = target - prediction

        # Step 3: Update weights and biases for both neurons
        # Neuron 0 updates
        #self.neurons[0].weights[0] += error * self.learning_rate * inp_0
        #self.neurons[0].weights[1] += error * self.learning_rate * inp_1
        self.neurons[0].weights[0] += error * self.learning_rate * self.normalizers[0]
        self.neurons[0].weights[1] += error * self.learning_rate * self.normalizers[1]

        #This was odd for Predict_Income_2_Inputs
        #self.normalizers[0] += error * self.learning_rate * inp_0
        #self.normalizers[1] += error * self.learning_rate * inp_1
        self.neurons[0].bias += error * self.learning_rate

        # Neuron 1 updates
        self.neurons[1].weights[0] += error * self.learning_rate * inp_0
        self.neurons[1].weights[1] += error * self.learning_rate * inp_1
        self.neurons[1].bias += error * self.learning_rate

        return prediction
