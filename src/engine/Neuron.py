from src.Legos.ActivationFunctions import *


class Neuron:
    """
    Represents a single neuron with weights, bias, and an activation function.
    """
    layers = []             # Shared across all Gladiators, needs resetting per run
    neurons = []           # Shared across all Gladiators, needs resetting per run NOT INTENDED TO  BE PROTECTED
    output_neuron = None  #Shared access directly to the output neuron.
    # DELETE ME _output_neuron
    def __init__(self, nid: int, num_of_weights: int, learning_rate: float, weight_initializer, layer_id: int = 0, activation = None):
        #print(f"creating neuron - nid={nid}")
        self.nid                = nid
        self.layer_id           = layer_id  # Add layer_id to identify which layer the neuron belongs to
        self.num_of_weights     = num_of_weights
        #self.learning_rate     = learning_rate
        self.learning_rates     = [learning_rate] * (num_of_weights + 1)   # going to stick bias LR in element 0 even though it offsets all the indexes by 1
        self.raw_sum            = 0.0
        self.dead_counter       = 0
        self.activation_value   = 0.0

        #self.activation        =  activation or Activation_Linear         # function
        self.activation_gradient= 0.0  # Store activation gradient from forward pass
        self.error_signal       = 1111.11
        self.weight_adjustments = ""        #TODO get rid of this
        self.blame_calculations = ""        #TODO get rid of this


        # ✅ Apply weight initializer strategy
        self.initialize_weights(weight_initializer)
        self.neuron_inputs = np.zeros_like(self.weights)

        # ✅ Ensure activation is never None
        self.activation = activation if activation is not None else Activation_NoDamnFunction
        self.activation_name = self.activation.name  # ✅ No more AttributeError

        # add to API collection
        Neuron.neurons.append(self)

        # Ensure layers list is large enough to accommodate this layer_id
        while len(Neuron.layers) <= layer_id:
            Neuron.layers.append([])

        # Add neuron to the appropriate layer and set its position
        Neuron.layers[layer_id].append(self)
        self.position = len(Neuron.layers[layer_id]) - 1  # Zero-based position within layer

        # If this neuron is in the last layer, set it as the output neuron
        if layer_id == len(Neuron.layers) - 1:
            Neuron.output_neuron = self  # Assign this neuron as the output neuron

    def initialize_weights(self, weight_initializer):
        self.weights, self.bias = weight_initializer((self.num_of_weights,))
        self.weights_before = self.weights.copy()
        self.bias_before = self.bias


        # 🔹 Adam optimizer state (momentum and RMS terms)
        total_params                    = self.num_of_weights + 1  # +1 for bias
        self.m                          = [0.0] * total_params
        self.v                          = [0.0] * total_params
        self.accumulated_accepted_blame = [0.0] * total_params
        self.t                          = 0  # Not a list! Just an integer counter for the neuron - NOT each weight

    def reinitialize(self, new_initializer):
        """Reinitialize weights & bias using a different strategy."""
        #self.weight_initializer = new_initializer
        self.initialize_weights(new_initializer)

    def activate(self):
        """Applies the activation function."""
        self.activation_value = self.activation(self.raw_sum)
        self.activation_gradient = self.activation.apply_derivative(self.activation_value)  # Store gradient!
        #self.check_for_dead_relu() # ☠️ Dead ReLU Detection (Optional: toggle via config or debug flag

    """
    def check_for_dead_relu(self, dead_threshold = 10):
        " ""☠️ Tracks persistent ReLU death before triggering reset."" "
        if self.activation != Activation_ReLU:
            if self.activation_value != 0:
                if abs(self.error_signal) > 1e-6:
                    self.dead_counter = 0           # 🎉 Still alive
                    return                          # 🎉 Still alive

        #print(f"checking if N{self.nid} is dead - dead_counter={self.dead_counter}")
        self.dead_counter += 1
        if self.dead_counter >= dead_threshold:
            self.maybe_soft_reset()



    def maybe_soft_reset(self):
        " ""💀 Optional: Soft-reset weights if ReLU is completely dead." ""
        if (
            self.activation_name == "ReLU"
            and self.activation_value == 0
            and abs(self.error_signal) < 1e-6
        ):
            print(f"🔁 Soft-resetting dead neuron {self.nid}")
            # Option A: Small nudge
            #self.weights += np.random.uniform(-0.01, 0.01, size=self.weights.shape)
            #self.bias += np.random.uniform(-0.01, 0.01)


            # Option B: Stronger nudge (helps break death loop)
            self.weights += np.random.uniform(0.5, 1.0, size=self.weights.shape)
            self.bias += np.random.uniform(0.5, 1.0)

            # Option C: Full reinit (if you'd rather go nuclear)
            # self.weights = np.random.uniform(-0.5, 0.5, size=self.weights.shape)
            # self.bias = np.random.uniform(-0.5, 0.5)

            self.dead_counter = 0
            """


    def set_activation(self, activation_function):
        """Dynamically update the activation function."""
        self.activation = activation_function
        self.activation_name = activation_function.name

    def compute_gradient(self):
        """Use the derivative for backpropagation."""
        return self.activation.apply_derivative(self.activation_value)

    def set_learning_rate(self, value):
        # Replace the existing learning_rates list with one where every element is the new value.
        self.learning_rates = [value] * (self.num_of_weights + 1)# * len(self.learning_rates)


    def learning_rate_report(self,extra_msg : str = "" ):
        print(f"LEARNING RATE REPORT FOR NEURON:{self.nid}\t {extra_msg} ")
        print(self.learning_rates)

    @classmethod
    def reset_layers(cls):
        """ Clears layers before starting a new Gladiator. """
        cls.layers.clear()
        cls.output_neuron = None
        cls._neurons = None


    @staticmethod
    def bulk_insert_weights(db, model_id, epoch, iteration):
        """
        Collects all weight values across neurons and creates a bulk insert SQL statement.
        """
        sql_statements = []

        # Ensure model_id is wrapped in single quotes if it's a string
        model_id_str = f"'{model_id}'" if isinstance(model_id, str) else model_id

        for layer in Neuron.layers:
            for neuron in layer:
                for weight_id, (prev_weight, weight ) in enumerate(zip(neuron.weights_before, neuron.weights)):
                    sql_statements.append(
                        f"({model_id_str}, {epoch}, {iteration}, {neuron.nid}, {weight_id + 1}, {prev_weight}, {weight})"
                    )
                #if neuron.nid==0: #DELETEME
                #    print(f"Epoch={neuron.bias_before}\tItearation={neuron.bias_before}\tBias Before={neuron.bias_before}\tBIAS AFTER={neuron.bias}")
                # Store bias separately as weight_id = 0
                sql_statements.append(
                    f"({model_id_str}, {epoch}, {iteration}, {neuron.nid}, 0, {neuron.bias_before}, {neuron.bias})"
                )

        if sql_statements:
            sql_query = f"INSERT INTO Weight (model_id, epoch, iteration, nid, weight_id, value_before, value) VALUES {', '.join(sql_statements)};"
            #print(f"Query for weights: {sql_query}")
            db.execute(sql_query)

