    ################################################################################################
    ################################ SECTION 1 - Training Default Methods ##########################
    ################################################################################################

    def forward_pass(self, training_sample):
        """
        Computes forward pass for each neuron in the XOR MLP.
        """
        #print("🚀Using Default Forward pass - to customize override forward_pass")
        input_values = training_sample[:-1]

        # 🚀 Compute raw sums + activations for each layer
        for layer_idx, layer in enumerate(Neuron.layers):  # Exclude output layer
            prev_activations = input_values if layer_idx == 0 else [n.activation_value for n in Neuron.layers[layer_idx - 1]]

            for neuron in layer:
                neuron.raw_sum = sum(input_val * weight for input_val, weight in zip(prev_activations, neuron.weights))
                neuron.raw_sum += neuron.bias
                neuron.activate()


    def back_pass(self, training_sample, loss_gradient: float):
        #print("🚀Using Default back_pass - to customize override back_pass")
        output_neuron = Neuron.layers[-1][0]

        # Step 1: Compute error signal for output neuron
        self.back_pass__error_signal_for_output(loss_gradient)

        # Step 2: Compute error signals for hidden neurons
        # * MUST go in reverse order!
        # * MUST be based on weights BEFORE they are updated.(weight as it was during forward prop
        for layer_index in range(len(Neuron.layers) - 2, -1, -1):  # Exclude output layer
            for hidden_neuron in Neuron.layers[layer_index]:  # Iterate over current hidden layer
                self.back_pass__error_signal_for_hidden(hidden_neuron)

        # Step 3: Adjust weights for the output neuron
        prev_layer_activations = [n.activation_value for n in Neuron.layers[-2]]  # Last hidden layer activations
        self.back_pass__distribute_error(output_neuron, prev_layer_activations)

        # Step 4: Adjust weights for the hidden neurons (⬅️ Last step we need)
        for layer_index in range(len(Neuron.layers) - 2, -1, -1):  # Iterate backwards (including first hidden layer)
            prev_layer_activations = [n.activation_value for n in Neuron.layers[layer_index - 1]]  # Use activations for hidden layers
            if layer_index == 0:        #For layer zero overwrite prev_layer_activations with inputs as inputs aren't in the neuron layers.
                prev_layer_activations = training_sample[:-1]  # Use raw inputs for first hidden layer
            for neuron in Neuron.layers[layer_index]:
                self.back_pass__distribute_error(neuron, prev_layer_activations)

    def back_pass__error_signal_for_output(self, loss_gradient: float):
        """
        Calculate error_signal(gradient) for output neuron.
        Assumes one output neuron and that loss_gradient has already been calculated.
        """
        output_neuron               = Neuron.layers[-1][0]
        activation_gradient         = output_neuron.activation_gradient
        error_signal                = loss_gradient * activation_gradient
        output_neuron.error_signal  = error_signal

    def back_pass__error_signal_for_hidden(self, neuron: Neuron):
        """
        Calculate the error signal for a hidden neuron by summing the contributions from all neurons in the next layer.
        args: neuron:  The neuron we are calculating the error for.
        """
        activation_gradient = neuron.activation_gradient
        total_backprop_error = 0  # Sum of (next neuron error * connecting weight)
        neuron.blame_calculations=""

        #print(f"Calculating error signal epoch/iter:{self.epoch}/{self.iteration} for neuron {to_neuron.layer_id},{to_neuron.position}")
        # 🔄 Loop through each neuron in the next layer

        memory_efficent_way_to_store_calcs = []
        for next_neuron in Neuron.layers[neuron.layer_id + 1]:  # Next layer neurons
            #print (f"
            # getting weight and error from {to_neuron.layer_id},{to_neuron.position}")
            weight_to_next = next_neuron.weights_before[neuron.position]  # Connection weight #TODO is weights before requried here?  I dont think so
            error_from_next = next_neuron.error_signal  # Next neuron’s error signal
            total_backprop_error += weight_to_next * error_from_next  # Accumulate contributions
            #OLD WAY neuron.blame_calculations= neuron.blame_calculations + f"{smart_format( weight_to_next)}!{smart_format( error_from_next)}@"
            memory_efficent_way_to_store_calcs.append(f"{smart_format(weight_to_next)}!{smart_format(error_from_next)}@")
        neuron.blame_calculations = ''.join(memory_efficent_way_to_store_calcs)  # Join once instead of multiple string concatenations


        # 🔥 Compute final error signal for this hidden neuron
        neuron.error_signal = activation_gradient * total_backprop_error


    def back_pass__distribute_error(self, neuron: Neuron, prev_layer_values):
        """
        Updates weights for a neuron based on error signal.
        args: neuron: The neuron that will have its weights updated to.

        - First hidden layer uses inputs from training data.
        - All other neurons use activations from the previous layer.
        """
        learning_rate = neuron.learning_rate
        error_signal = neuron.error_signal
        weight_formulas = []
        #if neuron.nid    == 2 and self.epoch==1 and self.iteration<3:
        #print(f"WEIGHT UPDATE FOR epoch:{self.epoch}\tItertion{self.iteration}")

        for i, (w, prev_value) in enumerate(zip(neuron.weights, prev_layer_values)):
            weight_before = neuron.weights[i]

            neuron.weights[i] += learning_rate * error_signal * prev_value
            if neuron.nid    == 2 and self.epoch==0 and self.iteration==0:
                print(f"w={w}\tneuron.weights[i]={neuron.weights[i]}\tneuron.weights_before[i]={neuron.weights_before[i]}")
                #print(f"weight# {i}  weight_before={smart_format(weight_before)}\tlearning_rate={learning_rate}\terror_signal={smart_format(error_signal)}\tprev_value={prev_value}\tnew weight={smart_format(neuron.weights[i])}\t")
            neuron_id = f"{neuron.layer_id},{neuron.position}"
            calculation = f"w{i} Neuron ID{neuron_id} = {store_num(w)} + {store_num(learning_rate)} * {store_num(error_signal)} * {store_num(prev_value)}"
            weight_formulas.append(calculation)
        if neuron.nid    == 2 and self.epoch==0 and self.iteration==0:
            print (f"All weights for neuron #{neuron.nid} epoch:  {self.epoch}\tItertion{self.iteration}\tWeights before=>{neuron.weights_before}\t THEY SHOULD BE -0.24442546 -0.704763 #Weights before=>{neuron.weights}")
        # Bias update
        neuron.bias += learning_rate * error_signal
        weight_formulas.append(f"B = {store_num(neuron.bias_before)} + {store_num(learning_rate)} * {store_num(error_signal)}")
        neuron.weight_adjustments = '\n'.join(weight_formulas)


    def validate_pass(self, target: float, prediction_raw:float):
        error = target - prediction_raw

        #loss = self.config.loss_function(prediction_raw, target)  # Uses selected loss function
        #loss_gradient = self.config.loss_function.grad(prediction_raw, target)  # Uses loss derivative
        loss = error ** 2  # Example loss calculation (MSE for a single sample)
        loss_gradient = error * 2 #For MSE it is linear.
        prediction =  1 if prediction_raw > .5 else 0      # Apply step function

        return error, loss, prediction, loss_gradient


