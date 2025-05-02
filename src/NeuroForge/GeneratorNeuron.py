#from src.NeuroForge.DisplayModel import DisplayModel
from src.NeuroForge.DisplayModel__Neuron import DisplayModel__Neuron
from src.engine.Config import Config


class GeneratorNeuron:
    model = None #Refernce to the Model it is creating neurons for.

    @staticmethod
    def create_neurons(the_model, max_act: float, margin=20, gap=30, max_neuron_size=350):
        """
        Create neuron objects, dynamically positioning them based on architecture.

        Parameters:
            the_model(DisplayModel): Reference to model the neurons are being created for.
            margin (int): Space around the entire neuron visualization.
            gap (int): Minimum space between neurons (both horizontally and vertically).
            max_neuron_size (int): Maximum allowed size for a single neuron.
        """
        # 🔹 Compute neuron size and spacing
        GeneratorNeuron.model = the_model
        #print(f"GeneratorNeuron.model = {GeneratorNeuron.model }")
        #print(f"Before calculate_neuron_size: margin={margin}, gap={gap}, max_neuron_size={max_neuron_size}")

        size = GeneratorNeuron.calculate_neuron_size( margin, gap, max_neuron_size)
        #print(f"Neuron size:{size}")
        max_neurons = max(GeneratorNeuron.model.config.architecture)  # Determine the layer with the most neurons
        max_layers = len(GeneratorNeuron.model.config.architecture)  # Total number of layers
        print(f"size={size}")
        #text_version = "Concise" if max(max_neurons,max_layers)  > 3 else "Verbose" # Choose appropriate text method based on network size
        text_version = "Concise" if size > 150 else "Verbose" # Choose appropriate text method based on network size
        width_needed = size * max_layers + (max_layers -1) * gap + margin * 2
        extra_width = GeneratorNeuron.model.width - width_needed
        extra_width_to_center = extra_width / 2

        #print(f"height_needed={height_needed}\tself.height={self.height}\textra_height={extra_height}\textra_height_to_center={extra_height_to_center}")
        # 🔹 Compute the horizontal centering offset (adjust for EZSurface width)
        offset = (GeneratorNeuron.model.screen_width - GeneratorNeuron.model.width)
        #print(f"self.screen_width={self.screen_width}\tself.width={self.width}\toffset={offset}")

        # 🔹 Create neurons
        GeneratorNeuron.model.neurons = []
        nid = -1

        #print (f"architecture = {GeneratorNeuron.model.config.architecture}")
        for layer_index, neuron_count in enumerate(GeneratorNeuron.model.config.architecture):

            if layer_index == len(GeneratorNeuron.model.config.architecture) - 1: #Add graph if last layer
                neuron_count += 1                                   # Inject 1 extra neuron in the last layer to hold graph

            layer_neurons = []

            # 🔹 Compute X coordinate for neurons in this layer
            x_coord = size * layer_index + layer_index * gap  + margin + extra_width_to_center

            for neuron_index in range(neuron_count):
                nid += 1  # Increment neuron ID
                height_needed = size * neuron_count + (neuron_count -1) * gap
                extra_height = GeneratorNeuron.model.height - height_needed
                extra_height_to_center = extra_height  // 2
                y_coord = size * neuron_index + gap *   neuron_index + margin/696969 + extra_height_to_center

                # 🔹 Instantiate Neuron (DisplayModel)
                #print(f"GENERATING NEURON 1 =  left={x_coord}, top={y_coord}")
                neuron = DisplayModel__Neuron( GeneratorNeuron.model, left=x_coord, top=y_coord, width=size, height=size, nid=nid, layer=layer_index, position=neuron_index, output_layer=len(GeneratorNeuron.model.config.architecture)-1, text_version=text_version, model_id=GeneratorNeuron.model.config.gladiator_name, screen=the_model.surface, max_activation=max_act )
                layer_neurons.append(neuron)
            GeneratorNeuron.model.neurons.append(layer_neurons)
        GeneratorNeuron.separate_graph_holder_from_neurons()
    @staticmethod
    def separate_graph_holder_from_neurons():
        graph_slot = GeneratorNeuron.model.neurons[-1][-1]  # Last neuron in last layer
        GeneratorNeuron.model.graph_holder = graph_slot  # or .graph_panel, .graph_target, etc.
        GeneratorNeuron.model.neurons[-1].pop()  # Removes it from draw loop if needed

    @staticmethod
    def calculate_neuron_size( margin, gap, max_neuron_size):
        """
        Calculate the largest neuron size possible while ensuring all neurons fit
        within the given surface dimensions.

        Parameters:

            margin (int): Space around the entire neuron visualization.
            gap (int): Minimum space between neurons (both horizontally & vertically).
            max_neuron_size (int): Max allowable size per neuron.

        Returns:
            int: Optimized neuron size.
        """
        #print(f"Inside calculate_neuron_size: margin={margin}, gap={gap}, max_neuron_size={max_neuron_size}")
        # Force a minimum of 2 neurons per layer for display (even if the model has only 1)
        padded_architecture = [max(2, n) for n in GeneratorNeuron.model.config.architecture]
        max_neurons = max(padded_architecture)  # Determine the layer with the most neurons
        max_layers = len(GeneratorNeuron.model.config.architecture)  # Total number of layers

        # 🔹 Compute maximum available height and width
        available_height = GeneratorNeuron.model.height - (2 * margin)   - ( max_neurons - 1) * gap
        available_width = GeneratorNeuron.model.width - (2 * margin)        - ( max_layers - 1) * gap
        #print(f"available_width ={available_width}")
        width_per_cell = available_width // max_layers
        #print (f"width_per_cell={width_per_cell}")
        height_per_cell = available_height // max_neurons
        # 🔹 Take the minimum size that fits both width and height constraints
        optimal_neuron_size = min(width_per_cell, height_per_cell, max_neuron_size)
        #print (f"optimal_neuron_size={optimal_neuron_size}")
        return optimal_neuron_size













    @staticmethod
    def NOTINUSEposition_neurons(self, margin=20, gap=60, max_neuron_size=400): #TODO try this out instead of original
        """Dynamically position neurons within this model's assigned space."""

        architecture = self.architecture[1:]  # Skip input layer (we only display neurons)
        max_neurons = max(architecture)  # Most neurons in any single layer
        max_layers = len(architecture)  # Total number of layers

        # Compute neuron size to fit within the model area
        size = self.calculate_neuron_size(margin, gap, max_neuron_size)

        width_needed = size * max_layers + (max_layers - 1) * gap + margin * 2
        height_needed = size * max_neurons + (max_neurons - 1) * gap + margin * 2

        extra_width_to_center = (self.width - width_needed) / 2
        extra_height_to_center = (self.height - height_needed) / 2

        # Assign neurons to their respective layers
        self.neurons = []
        nid = -1  # Unique neuron ID

        for layer_index, neuron_count in enumerate(architecture):
            layer_neurons = []

            x_coord = size * layer_index + layer_index * gap + margin + extra_width_to_center

            for neuron_index in range(neuron_count):
                nid += 1
                y_coord = size * neuron_index + gap * neuron_index + margin + extra_height_to_center
                print("GENERATING NEURON 2")
                neuron = DisplayModel__Neuron(
                    nid=nid, layer=layer_index, position=neuron_index,
                    output_layer=len(architecture) - 1, db=Const.dm.db, model_id=self.model_id
                )

                neuron.location_left = x_coord
                neuron.location_top = y_coord
                neuron.location_width = size
                neuron.location_height = size

                layer_neurons.append(neuron)

            self.neurons.append(layer_neurons)

        #print(f"✅ Positioned {nid + 1} neurons for model {self.model_id}.")

"""
    def add_input_connections(self, forward: bool):
        " ""Create connections from the left edge to the first hidden layer." ""
        first_hidden_layer = self.neurons[0]
        if not first_hidden_layer:
            return
        origin_point = (0, self.height // 2)
        for neuron in first_hidden_layer:
            self.connections.append(DisplayModel__ConnectionForward(from_neuron=origin_point, to_neuron=neuron))

    def add_output_connections(self, forward: bool):
        "" "Create connections from last output neuron to prediction box."" "
        dest_point = (self.width, 144)
        self.connections.append(DisplayModel__ConnectionForward(from_neuron=self.neurons[-1][0], to_neuron=dest_point))

"""