from src.engine.BaseGladiator import Gladiator
import math

class SuzukiHayabusa_XOR(Gladiator):
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
        self.initialize_neurons(3)
        #self.normalizers = self.training_data.normalizers  # Output: [0.333, 0.666]
        #self.training_data.set_normalization_min_max()

        #Play with intial values
        self.neurons[0].weights[0]  = .25
        self.neurons[0].weights[1]  = .75
        self.neurons[1].weights[0]  = 0.5
        self.neurons[1].weights[1]  = 0.5
        self.neurons[2].weights[0]  = 1
        self.neurons[2].weights[1]  = 1
        self.neurons[0].bias        = 0
        self.neurons[1].bias        = 0
        self.neurons[2].bias        = 0



    def training_iteration(self, training_data) -> float:
        input_1 = training_data[0]  # First input
        input_2 = training_data[1]  # Second input
        target = training_data[-1]  # Target value

        prediction = self.forward_pass(input_1, input_2)
        # Step 5: Calculate the error and loss
        error = target - prediction
        loss = error ** 2  # Mean Squared Error Loss function
        print (f"Error and Loss *****************\terror={error}\tloss={loss}")
        return  prediction
        backwards_pass()

    def backwards_pass():
        """
        Steps to backprop
        1. Compute the Error for Output Neuron - MSE is always derivative of 2
            The loss is already calculated as squared error:
            d_loss_d_out = 2 * error  # Derivative of MSE w.r.t. output
        2. Gradient for Output Neuron
            d_out_raw = d_loss_d_out * (1 - out_tanhed ** 2)
        3. Backpropagate to Hidden Neurons - Gradient for hidden neuron
            d_h1_raw = d_out_raw * self.neurons[2].weights[1] * (1 - h1_output ** 2)
            d_h0_raw = d_out_raw * self.neurons[2].weights[0] * (1 - h0_output ** 2)
        4. Update the weights and biases for the hidden neurons:
            # Hidden neuron H1
            self.neurons[1].weights[0] += self.learning_rate * d_h1_raw * inp_0
            self.neurons[1].weights[1] += self.learning_rate * d_h1_raw * inp_1
            self.neurons[1].bias += self.learning_rate * d_h1_raw

            ATTEMPT 2 to document backprop!

                1) Get the derivative of the loss wrt the output.  It is a function of which loss function is used.  MSE is always 2.  BCE would be different.
                2) Get the Gradient of the output.  for this i need
                A)  The derivative above
                B) a value from forward prop, 'out_tanhed'(so i better save it)
                Now i have what i need to update the weight and bias of the output neuron.
                3)  Update hidden neurons.  Start by calculating their gradient.  this requires
                A) Derivative of output neuron d_out_raw
                B) Weight itself.
                c) Neurons activation (so again i better save it)
                4) Finally adjust the weights of hidden neurons by
                ... LR* gradient(same for all weights within neuron)  different between neurons * input that goes with that weight.
        """
        """
        H0 raw H01*i1 +H02*i2
            H01 = .25 * 1 = .25
            HO2 = .75 * 0
       
       Key Suggestions for Variable Names:
        Inputs and Outputs:
            Use input_1, input_2 for raw input values.
            Use hidden_1_raw, hidden_2_raw for pre-activation values.
            Use hidden_1_output, hidden_2_output for post-activation values.
        Weights and Biases:        
        Use hidden_1_weight_1, hidden_1_weight_2, and hidden_1_bias for H1's weights and bias.
        Similarly, output_weight_h1, output_weight_h2, and output_bias for the output neuron.
        Intermediate Calculations:
        
        Use output_raw for the sum of weighted inputs to the output neuron.
        Use output_activated for the tanh-activated output value.
        Prediction:
        
        Use prediction_step for the step function output (binary 0 or 1 for classification).
        """

    def forward_pass(self, input_1, input_2) -> float:
        """
        :param inputs from I1 and I2
        :return prediction
        Hello MLP world!
        """
        print (f"STARTING Iteration FORWARD PASS*****************\tinput_1={input_1}\tinput_2={input_2}\t")
        # Step 1: Compute the output of the first hidden neuron
        hidden_1_raw    =( input_1 * self.neurons[0].weights[0] +
                           input_2 * self.neurons[0].weights[1] +
                           self.neurons[0].bias)
        hidden_1_output =  self.tanh(hidden_1_raw)
        print(f"hidden_1_raw===( {input_1} * {self.neurons[0].weights[0]} +{input_2} * {self.neurons[0].weights[1]} +{self.neurons[0].bias} = {hidden_1_raw} <==DOES IT???)")
        print (f"Tanh({hidden_1_raw})={hidden_1_output}")

        # Step 2: Compute the output of the second hidden neuron
        hidden_2_raw    =( input_1 * self.neurons[1].weights[0] +
                           input_2 * self.neurons[1].weights[1] +
                           self.neurons[1].bias)
        hidden_2_output =  self.tanh(hidden_2_raw)
        print(f"hidden_2_raw===({input_1} * {self.neurons[1].weights[0]} +{input_2} * {self.neurons[1].weights[1]} +{self.neurons[1].bias} = {hidden_2_raw} <==DOES IT???)")
        print (f"Tanh({hidden_2_raw})={hidden_2_output}")

        # Step 3: Compute the output of the output neuron - that is the predicton
        output_raw      =( hidden_1_output * self.neurons[2].weights[0] +
                           hidden_2_output * self.neurons[2].weights[1] +
                           self.neurons[2].bias)
        output_tanhD    =  self.tanh(output_raw)              # Apply tahn function
        prediction_step =  1 if output_tanhD > 0 else 0      # Apply step function
        print(f"output_raw===({hidden_1_output} * {self.neurons[2].weights[0]} +{hidden_2_output} * {self.neurons[2].weights[1]} +{self.neurons[2].bias} = {output_raw} <==DOES IT???)")
        print (f"Tanh({output_raw})={output_tanhD}\tprediction_step={prediction_step}")
        print (f"FORWARD PASS complete *************************************************************")
        return prediction_step




    def tanh(self,x: float):
        """
        Compute the hyperbolic tangent of x.

        Args:
            x (float): The input value.

        Returns:
            float: The hyperbolic tangent of the input.
        """
        # Logic - but not optimized ==>return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        return math.tanh(x)

        # SLP for updating neurons
        #print(f"in hayabusatwoneurontron Before: self.neurons[0].weights[0]={self.neurons[0].weights[0]}\tself.neurons[0].weights[1]={self.neurons[0].weights[1]}" )
        # Step 3: Update weights and biases for the first neuron
        #self.neurons[0].weights[0] += error * self.learning_rate * self.normalizers[0]
        #self.neurons[0].weights[1] += error * self.learning_rate * self.normalizers[1]
        #self.neurons[0].bias += error * self.learning_rate

        # Step 4: Update weights and biases for the first neuron
        #self.neurons[1].weights[0] += error * self.learning_rate * self.normalizers[0]
        #self.neurons[1].weights[1] += error * self.learning_rate * self.normalizers[1]
        #self.neurons[1].bias += error * self.learning_rate
        #print(f"in hayabusatwoneurontron AFTER: self.neurons[0].weights[0]={self.neurons[0].weights[0]}\tself.neurons[0].weights[1]={self.neurons[0].weights[1]}" )
        # (Second neuron deliberately ignored)