#from src.NeuroForge.DisplayModel import DisplayModel
from src.Legos.Scalers import Scaler_NONE
from src.NeuroForge.DisplayModel__Neuron import DisplayModel__Neuron
from src.NeuroForge.DisplayModel__NeuronScaler import DisplayModel__NeuronScaler
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

        size = GeneratorNeuron.calculate_neuron_size(margin, gap, max_neuron_size)
        #print(f"Neuron size:{size}")
        max_neurons = max(GeneratorNeuron.model.config.architecture)  # Determine the layer with the most neurons
        max_layers = len(GeneratorNeuron.model.config.architecture)  # Total number of layers
        #print(f"size={size}")
        text_version = "Concise" if size < 350 else "Verbose" # Choose appropriate text method based on network size

        # Leave space for input scaler if there is one.
        scaler_offset = 0 if GeneratorNeuron.model.config.scaler.inputs_are_unscaled else 1
        max_layers += scaler_offset

        # old horizontal positioning logic
        width_needed = size * max_layers + (max_layers -1) * gap + margin * 2
        extra_width = GeneratorNeuron.model.width - width_needed
        extra_width_to_center = extra_width / 2

        # 🔹 Compute dynamic x positions for each layer (including scaler column)
        x_positions = GeneratorNeuron._compute_layer_x_positions(
            size=size,
            margin=margin,
            total_width=GeneratorNeuron.model.width,
            min_gap=gap,
            num_columns=max_layers
        )

        # 🔹 Create neurons
        GeneratorNeuron.model.neurons = []
        nid = -1

        #GeneratorNeuron.maybe_add_input_scaler_visual(size, margin, extra_width_to_center, text_version, max_act, gap)
        GeneratorNeuron.maybe_add_input_scaler_visual(
            size, text_version, max_act, x_positions
        )
        #print (f"architecture = {GeneratorNeuron.model.config.architecture}")
        for layer_index, neuron_count in enumerate(GeneratorNeuron.model.config.architecture):
            # Inject 1 extra neuron in the last layer to hold graph
            if layer_index == len(GeneratorNeuron.model.config.architecture) - 1:
                neuron_count += 1

            layer_neurons = []

            # 🔹 Compute X coordinate for neurons in this layer (evenly spread)
            x_coord = x_positions[layer_index + scaler_offset]
            #print(f"total horizontal space =  {GeneratorNeuron.model.width}")


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
    def maybe_add_input_scaler_visual(
        size: int,
        text_version: str,
        max_act: float,
        x_positions: list[float]
    ):
        """
        Optionally add a visual neuron for the input scaler,
        pinned at the very first column (x_positions[0]).
        """
        model = GeneratorNeuron.model
        if model.config.scaler.inputs_are_unscaled:
            model.input_scaler_neuron = None
            return  # nothing to show

        # first column is where our scaler sits
        x_coord = x_positions[0]
        y_coord = model.height // 2 - size // 2

        scaler_neuron = DisplayModel__NeuronScaler(
            model,
            left=x_coord,
            top=y_coord,
            width=size,
            height=size,
            nid=-1,         # Not in data flow
            layer=-1,
            position=0,
            output_layer=0,
            text_version=text_version,
            model_id=model.config.gladiator_name,
            screen=model.surface,
            max_activation=max_act
        )
        model.input_scaler_neuron = scaler_neuron

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

        # Add input scalar if needed
        #if GeneratorNeuron.model.config.input_scaler != Scaler_NONE: # and 1 == 2:
        if GeneratorNeuron.model.config.scaler.inputs_are_unscaled:
            padded_architecture.insert(0, 1)

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
    def _compute_layer_x_positions(
        size: int,
        margin: int,
        total_width: int,
        min_gap: float,
        num_columns: int
    ) -> list[float]:
        """
        Return a list of x-coordinates for each column so that:
          • all (N+1) gaps (left edge → col0, between cols, colN-1 → right edge)
            are equal if possible,
          • but never smaller than min_gap.
        """
        # center a single column
        if num_columns < 2:
            center = (total_width - size) / 2
            return [margin + center]

        # raw gap required for perfect equal-spacing
        raw = (total_width - 2*margin - num_columns*size) / (num_columns + 1)
        gap = max(raw, min_gap)

        # now build positions: first column sits at margin + gap
        return [
            margin + gap + i * (size + gap)
            for i in range(num_columns)
        ]
