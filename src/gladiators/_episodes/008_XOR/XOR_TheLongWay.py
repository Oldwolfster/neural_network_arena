import numpy as np

from src.engine.ActivationFunction import Tanh, Sigmoid
from src.engine.BaseGladiator import Gladiator
import math

class XOR_TheLongWay(Gladiator):
    """
    This is my foray into MLP (Multi-Layer Perceptron) by solving the XOR problem,
    which cannot be linearly separated; hence, an SLP (Single-Layer Perceptron) just won't do.

    FORWARD PASS:
    Often, it is depicted as 5 neurons ([] below depicts a neuron):
        [Input1] | Both inputs connect | ==> \\    [Hidden1 (W11, W12, BiasH1)]   ===> [Output (WH1, WH2, B)]
        [Input2] | to both weights     | ==> //    [Hidden2 (W21, W22, BiasH2)]   ==/

    However, inputs are just part of the sample data, not really neurons themselves.
    So, we will declare 3 neurons: H1, H2, and O (hidden and output layers).

    Each hidden neuron (H1, H2) will be calculated as:
        tanh(W1 * Input1 + W2 * Input2 + BiasH1)

    The output neuron (O) will also apply the tanh activation function,
    but the final output will use a step function: if > 0, predict True; else, predict False.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.initialize_neurons([2])

        """ Below is example of weights that make the model work right away
        # Hidden Layer 1
        self.neurons[0].weights[0] = 20.0  # Strong positive weight for first input
        self.neurons[0].weights[1] = 20.0  # Strong positive weight for second input
        self.neurons[0].bias = -10.0       # Negative bias to act as threshold

        # Hidden Layer 2
        self.neurons[1].weights[0] = -20.0  # Strong negative weight for first input
        self.neurons[1].weights[1] = -20.0  # Strong negative weight for second input
        self.neurons[1].bias = 30.0         # Large positive bias

        # Output Layer
        self.neurons[2].weights[0] = 20.0   # Strong positive weight from H1
        self.neurons[2].weights[1] = 20.0   # Strong positive weight from H2
        self.neurons[2].bias = -10.0        # Negative bias
        """

        # Initialize weights with small, non-symmetric values
        self.neurons[0].weights[0] = 0.1
        self.neurons[0].weights[1] = 0.2
        self.neurons[0].bias = 1

        self.neurons[1].weights[0] = 0.3
        self.neurons[1].weights[1] = 0.4
        self.neurons[1].bias = 2

        self.neurons[2].weights[0] = 0.5
        self.neurons[2].weights[1] = 0.6
        self.neurons[2].bias = 3

        self.learning_rate = 0.1                # Or even higher like 0.5
        self.output_tanh            = 0         # Variables we need from forward prop to do back prop
        self.hidden_1_output        = 0         # Variables we need from forward prop to do back prop
        self.hidden_2_output        = 0         # Variables we need from forward prop to do back prop
        self.neurons[0].set_activation(Tanh)        # Set activation to Tanh on neuron 1,0
        self.neurons[1].set_activation(Tanh)        # Set activation to Tanh on neuron 1,1
        self.neurons[2].set_activation(Sigmoid)     # Set activation to Sig on neuron 2,0

    def training_iteration(self, training_data) -> float:
        input_0 = training_data[0]  # First input
        input_1 = training_data[1]  # Second input
        target = training_data[-1]  # Target value

        prediction = self.forward_pass(input_0, input_1)     #without step applied

        # Step 5: Calculate the error and loss
        error = target - prediction
        loss = error ** 2  # Mean Squared Error Loss function
        print (f"Error and Loss        ******* Prediction:{prediction}\tTarget:{target}\tError={error}\tLoss={loss}")
        print()
        self.backwards_pass(error, input_0, input_1)
        prediction_step =  1 if self.output_tanh > 0 else 0      # Apply step function
        #return  prediction_step

    def forward_pass(self, input_0, input_1) -> float:
        """
        Manually computes forward pass for each neuron in the XOR MLP.

        :param input_0: First input feature
        :param input_1: Second input feature
        :return: prediction (final output of the network)
        """

        # 🔹 Inputs are explicitly provided
        input_values = [input_0, input_1]

        # 🚀 Compute raw sums + activations for first hidden layer (2 neurons)
        self.neurons[0].raw_sum = (
            (input_values[0] * self.neurons[0].weights[0]) +
            (input_values[1] * self.neurons[0].weights[1]) +
            self.neurons[0].bias
        )
        self.neurons[0].activate()

        self.neurons[1].raw_sum = (
            (input_values[0] * self.neurons[1].weights[0]) +
            (input_values[1] * self.neurons[1].weights[1]) +
            self.neurons[1].bias
        )
        self.neurons[1].activate()

        # 🚀 Compute raw sum + activation for Output neuron (Sigmoid)
        self.neurons[2].raw_sum = (
            (self.neurons[0].activation_value * self.neurons[2].weights[0]) +
            (self.neurons[1].activation_value * self.neurons[2].weights[1]) +
            self.neurons[2].bias
        )
        self.neurons[2].activate()

        # DEBUGGING:
        print(f"Hidden Neuron 0: raw_sum={self.neurons[0].raw_sum}, activation_value={self.neurons[0].activation_value}")
        print(f"Hidden Neuron 1: raw_sum={self.neurons[1].raw_sum}, activation_value={self.neurons[1].activation_value}")
        output_raw =  self.neurons[0].activation_value  * self.neurons[2].weights[0]  + self.neurons[1].activation_value * self.neurons[2].weights[1] + self.neurons[2].bias
        print(f"output_raw===({self.neurons[0].activation_value} * {self.neurons[2].weights[0]} + {self.neurons[1].activation_value} * {self.neurons[2].weights[1]} + {self.neurons[2].bias} = {output_raw} <==DOES IT???)")
        print(f"Output Neuron: raw_sum={self.neurons[2].raw_sum}, activation_value={self.neurons[2].activation_value}")

        return self.neurons[2].activation_value  # 🚀 Final prediction



        """
        Steps to backprop
        1. Compute the Error for Output Neuron - MSE is always derivative of 2
            The loss is already calculated as squared error:
            der_loss_WRT_output = 2 * error  # Derivative of MSE w.r.t. output
        2. Gradient for Output Neuron
            grad_output_WRT_RawOutput = der_loss_WRT_output * (1 - out_tanhed ** 2)
        3. Backpropagate to Hidden Neurons - Gradient for hidden neuron
            grad_h1_raw_wrt_loss = grad_output_WRT_RawOutput * self.neurons[2].weights[1] * (1 - h1_output ** 2)
            d_h0_raw = grad_output_WRT_RawOutput * self.neurons[2].weights[0] * (1 - h0_output ** 2)
        4. Update the weights and biases for the hidden neurons:
            # Hidden neuron H1
            self.neurons[1].weights[0] += self.learning_rate * grad_h1_raw_wrt_loss * inp_0
            self.neurons[1].weights[1] += self.learning_rate * grad_h1_raw_wrt_loss * inp_1
            self.neurons[1].bias += self.learning_rate * grad_h1_raw_wrt_loss

            ATTEMPT 2 to document backprop!

                1) Get the derivative of the loss wrt the output.  It is a function of which loss function is used.  MSE is always 2.  BCE would be different.
                2) Get the Gradient of the output.  for this i need
                A)  The derivative above
                B) a value from forward prop, 'out_tanhed'(so i better save it)
                Now i have what i need to update the weight and bias of the output neuron.
                3)  Update hidden neurons.  Start by calculating their gradient.  this requires
                A) Derivative of output neuron grad_output_WRT_RawOutput
                B) Weight itself.
                c) Neurons activation (so again i better save it)
                4) Finally adjust the weights of hidden neurons by
                ... LR* gradient(same for all weights within neuron)  different between neurons * input that goes with that weight.
        """
    def backwards_pass(self, error: float, input_0: float, input_1: float) -> None:
        """
        Performs the backward pass (backpropagation) to update weights and biases.

        Parameters:
            error (float): The difference between target and prediction (target - prediction)
            input_0 (float): First input to the network
            input_1 (float): Second input to the network

        neurons[0] = first neuron of hidden layer
        neurons[1] = 2nd neuron of hidden layer
        neurons[2] = output neuron
        """
        # Derivative of loss (MSE) with respect to prediction
        # MSE = (target - prediction)^2
        # d/dx(MSE) = -2(target - prediction) = 2(prediction - target) = -2(error)        # Measures how much the loss changes if the prediction changes
        # For scalar outputs, the gradient reduces to a single derivative
        # as there's only one dimension of change to consider
        grad_loss_wrt_prediction = -2 * error
        # Add gradient clipping to prevent vanishing gradients
        grad_loss_wrt_prediction = np.clip(-2 * error, -1, 1)


        # Gradient of the output neuron's raw activation (pre-tanh) with respect to the loss
        # Combines grad_loss_wrt_prediction and the derivative of tanh at the current activation value
        # Measures how much the raw output of the output neuron contributes to the loss
        # This value is multiplied by the outputs of the hidden neurons to update the weights and bias of the output neuron
        grad_outputRaw_wrt_loss = grad_loss_wrt_prediction * (1 - self.output_tanh ** 2)

        # Backprop to Hidden Neurons  - Gradient for hidden neuron WRT to the loss  - Each connection from hidden to the output
        #hidden_neuron 1 goes to weight 0 on output neuron because we are using the chain rule to backprop through connection
        grad_h1_raw_wrt_loss = grad_outputRaw_wrt_loss * self.neurons[2].weights[0] * (1 - self.hidden_1_output ** 2) #For Tanh only
        grad_h2_raw_wrt_loss = grad_outputRaw_wrt_loss * self.neurons[2].weights[1] * (1 - self.hidden_2_output ** 2) #For Tanh only

        #Update the weights and bias for the output neuron:
        self.neurons[2].weights[0] += self.learning_rate * grad_outputRaw_wrt_loss * self.hidden_1_output
        self.neurons[2].weights[1] += self.learning_rate * grad_outputRaw_wrt_loss * self.hidden_2_output
        self.neurons[2].bias += self.learning_rate * grad_outputRaw_wrt_loss

        # Update the weights and biases for the hidden neurons:
        # Weight update formula: w = w + learning_rate * gradient * input
        #       where gradient is the gradient of the loss with respect to the raw activation
        #       and input is the input to that particular weight (either the original inputs or hidden layer outputs)

        # Hidden neuron 1
        self.neurons[0].weights[0] += self.learning_rate * grad_h1_raw_wrt_loss * input_0
        self.neurons[0].weights[1] += self.learning_rate * grad_h1_raw_wrt_loss * input_1
        self.neurons[0].bias += self.learning_rate * grad_h1_raw_wrt_loss

        # Hidden neuron H2
        self.neurons[1].weights[0] += self.learning_rate * grad_h2_raw_wrt_loss * input_0
        self.neurons[1].weights[1] += self.learning_rate * grad_h2_raw_wrt_loss * input_1
        self.neurons[1].bias += self.learning_rate * grad_h2_raw_wrt_loss
