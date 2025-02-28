    def back_pass__error_signal_for_hidden(self, neuron: Neuron):
        """
        Calculate the error signal for a hidden neuron by summing the contributions from all neurons in the next layer.
        Stores the calculations in a structured list for later insertion into the database.
        """
        activation_gradient = neuron.activation_gradient
        total_backprop_error = 0  # Sum of (next neuron error * connecting weight)

        for next_neuron in Neuron.layers[neuron.layer_id + 1]:  # Next layer neurons
            weight_to_next = next_neuron.weights_before[neuron.position]  # Connection weight
            error_from_next = next_neuron.error_signal  # Next neuron’s error signal
            total_backprop_error += weight_to_next * error_from_next  # Accumulate contributions

            # Store calculation step as a structured tuple
            self.error_signal_calcs.append([
                self.epoch, self.iteration, self.gladiator, f"{neuron.layer_id},{neuron.position}",
                weight_to_next, "*", error_from_next, "=", None, None, weight_to_next * error_from_next
            ])
        neuron.error_signal = activation_gradient * total_backprop_error  # Compute final error signal


    def back_pass__distribute_error(self, neuron: Neuron, prev_layer_values):
        """
        Updates weights for a neuron based on error signal.
        Stores calculations in a structured list for later insertion.
        """
        learning_rate = neuron.learning_rate
        error_signal = neuron.error_signal
        neuron_id = f"{neuron.layer_id},{neuron.position}"

        for i, (w, prev_value) in enumerate(zip(neuron.weights, prev_layer_values)):
            weight_before = neuron.weights[i]

            # 🔹 Update weight
            neuron.weights[i] += learning_rate * error_signal * prev_value

            # 🔹 Store structured calculation
            self.distribute_error_calcs.append([
                self.epoch, self.iteration, self.gladiator, neuron_id,
                weight_before, "+", learning_rate, "*", error_signal, "*", prev_value, neuron.weights[i]
            ])

        # 🔹 Bias update
        bias_before = neuron.bias
        neuron.bias += learning_rate * error_signal
        self.distribute_error_calcs.append([
            self.epoch, self.iteration, self.gladiator, neuron_id,
            bias_before, "+", learning_rate, "*", error_signal, None, None, neuron.bias
        ])

    def insert_error_signal_calcs(self):

        """
        Inserts all backprop calculations for the current iteration into the database.
        """
        sql = """
        INSERT INTO ErrorSignalCalcs 
        (epoch, iteration, model_id, neuron_id, arg_1, op_1, arg_2, op_2, arg_3, op_3, result)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        self.db.executemany(sql, self.error_signal_calcs)
        self.error_signal_calcs.clear()

    def insert_distribute_error_calcs(self):

        """
        Inserts all weight update calculations for the current iteration into the database.
        """
        sql = """
        INSERT INTO WeightUpdates 
        (epoch, iteration, model_id, neuron_id, weight_index, arg_1, op_1, arg_2, op_2, arg_3, op_3, result)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

        self.db.executemany(sql, self.distribute_error_calcs)
        self.distribute_error_calcs.clear()

