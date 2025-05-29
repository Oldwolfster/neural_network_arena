#from src.NeuroForge.DisplayModel import DisplayModel
from src.Legos.Scalers import Scaler_NONE
from src.NeuroForge.DisplayModel__Neuron import DisplayModel__Neuron
from src.NeuroForge.DisplayModel__NeuronScaler import DisplayModel__NeuronScaler
from src.engine.Config import Config
import copy
from src.engine.Utils import ez_debug
class GeneratorNeuron:
    model = None #Refernce to the Model it is creating neurons for.
    nid = 0
    special_layers_at_end = 0
    output_layer = 0
    @staticmethod
    def get_full_architecture(hidden_and_output):
        GeneratorNeuron.output_layer = len(GeneratorNeuron.model.config.architecture)-1
        full_architecture = copy.deepcopy(hidden_and_output)

        # Add space for the graph below the output neuron
        if full_architecture and full_architecture[-1] < 2:
            full_architecture[-1] = 2

        # Prepend 1 if inputs are unscaled
        if GeneratorNeuron.model.config.scaler.inputs_are_scaled:
            full_architecture.insert(0, -1)     #-1 indicates it's a "1 special neuron"
            GeneratorNeuron.output_layer += 1

        # Append 1 if target is unscaled
        if GeneratorNeuron.model.config.scaler.target_is_scaled:
            full_architecture.append(-1)#-1 indicates it's a "1 special neuron"
        return  full_architecture

    @staticmethod
    def calculate_column_width(layer_count, max_neuron_size, min_gap, available_width):
        gap_count   = layer_count + 1
        entire_width = layer_count * max_neuron_size + (min_gap* gap_count)
        layer_width = max_neuron_size                       #will be adjusted down if too big.
        gap_width = min_gap                                       # will be increased if needed to center
        if entire_width > available_width:                  # Neuron width must be smaller than max to fit.
            overrun = entire_width - available_width
            per_layer = overrun / layer_count
            layer_width -= per_layer
        else:                                               # extra space so pad gap.
            extra = available_width - entire_width
            per_gap = extra / gap_count
            gap_width += per_gap
        return  layer_width, gap_width

    @staticmethod
    def create_neurons(the_model, max_act: float, margin=20, min_gap=50, max_neuron_size=350):
        """
        Create neuron objects, dynamically positioning them based on architecture.

        Parameters:
            the_model(DisplayModel): Reference to model the neurons are being created for.
            margin (int): Space around the entire neuron visualization.
            gap (int): Minimum space between neurons (both horizontally and vertically).
            max_neuron_size (int): Maximum allowed size for a single neuron.
        """
        # 🔹 Compute neuron size and spacing
        # reset the static ID counter
        GeneratorNeuron.nid = 0

        # clear out any old layers
        GeneratorNeuron.model   = the_model
        available_height        = GeneratorNeuron.model.height # - (2 * margin)   - ( max_neurons - 1) * gap
        available_width         = GeneratorNeuron.model.width   #- (2 * margin)        - ( max_layers - 1) * gap
        full_architecture       = GeneratorNeuron.get_full_architecture(GeneratorNeuron.model.config.architecture)
        layer_count             = len(full_architecture)
        layer_width, gap_width  = GeneratorNeuron.calculate_column_width(layer_count, max_neuron_size, min_gap, available_width)

        nid = 0
        true_layer_index = 0            # this WILL NOT include 'special layers' like layer index does.


        # Arrange Neurons one layer at a time
        for layer_index, neuron_count in enumerate(full_architecture):
            x_position = layer_index * layer_width + ((layer_index+1) * gap_width)
            GeneratorNeuron.create_layer(x_position, layer_width, layer_index, neuron_count, max_neuron_size, min_gap, available_height, max_act, true_layer_index)
            nid += abs(neuron_count) #not in use - class level one works... this oneis buggy
            if neuron_count > 0: true_layer_index += 1
        GeneratorNeuron.separate_graph_holder_from_neurons()
        GeneratorNeuron.model.layer_width = layer_width
        #print(f" layer_width = {layer_width}")



    @staticmethod
    def create_layer(x_position, layer_width, layer_index, neuron_count, max_neuron_size, min_gap, available_height, max_act, true_layer_index):
        #print(f"creating layer {layer_index} with {neuron_count} neurons  - nid = {GeneratorNeuron.nid}")
        neuron_height, gap_width = (GeneratorNeuron.calculate_column_width
                                    (abs(neuron_count), max_neuron_size, min_gap, available_height))
        #print(f"GeneratorNeuron.model = {GeneratorNeuron.model }")
        #print(f"Before calculate_neuron_size: margin={margin}, gap={gap}, max_neuron_size={max_neuron_size}")
        text_version = "Concise" if layer_width < 350 else "Verbose" # Choose appropriate text method based on network size
        layer_neurons = []
        if neuron_count > 0: # - negative indicates "special neurons - sscalers or threholders
            for neuron_index in range(neuron_count):
                #ez_debug(neuron_count=neuron_count)
                y_position = neuron_index * neuron_height + ((neuron_index+1) * gap_width)
                #ez_debug(GeneratorNeuron_output_layer=GeneratorNeuron.output_layer)
                #if not GeneratorNeuron.output_layer==layer_index and neuron_index >0:  #don't create neuron for graph
                neuron = DisplayModel__Neuron( GeneratorNeuron.model, left=x_position, top=y_position, width=layer_width, height=neuron_height, nid=GeneratorNeuron.nid, layer=true_layer_index, position=neuron_index, output_layer=GeneratorNeuron.output_layer, text_version=text_version, run_id=GeneratorNeuron.model.run_id, screen=GeneratorNeuron.model.surface, max_activation=max_act)
                layer_neurons.append(neuron)
                GeneratorNeuron.nid += 1
            GeneratorNeuron.model.neurons.append(layer_neurons)
        else: # - negative indicates "special neurons - sscalers or threholders
            for neuron_index in range(0, neuron_count, -1):
                #ez_debug(neuron_count=neuron_count)
                y_position = abs(neuron_index) * neuron_height + ((abs(neuron_index)+1) * gap_width)
                is_input = True
                if layer_index > 0: is_input=False #sets label
                scaler_neuron = DisplayModel__NeuronScaler(
                    left=x_position,
                    top=y_position,
                    width=layer_width,
                    height=neuron_height,
                    nid=-1,         # Not in data flow
                    layer=-1,
                    position=abs(neuron_index),
                    output_layer=0,
                    text_version=text_version,
                    run_id=GeneratorNeuron.model.run_id,
                    my_model=GeneratorNeuron.model,
                    screen= GeneratorNeuron.model.surface, max_activation=max_act,
                    is_input=is_input

                )
                if layer_index== 0:
                    GeneratorNeuron.model.input_scaler_neuron = scaler_neuron
                else:
                    GeneratorNeuron.model.prediction_scaler_neuron = scaler_neuron


    @staticmethod
    def separate_graph_holder_from_neurons():
        graph_slot = GeneratorNeuron.model.neurons[-1].pop() # removes the graph from the data structure of neurons and records the reference
        GeneratorNeuron.model.graph_holder = graph_slot  # or .graph_panel, .graph_target, etc.







        #GeneratorNeuron.maybe_add_input_scaler_visual(size, margin, extra_width_to_center, text_version, max_act, gap)
        #GeneratorNeuron.maybe_add_input_scaler_visual(
        #    size, text_version, max_act, x_positions
        #)
        #GeneratorNeuron.maybe_add_prediction_scaler_visual(
        #    size, text_version, max_act, x_positions, max_layers
        #)




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
        print("Adding input scaler")
        # first column is where our scaler sits
        x_coord = x_positions[0]
        y_coord = model.height // 2 - size // 2
        print(f"input scaler x_coord = {x_coord}")
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
    def maybe_add_prediction_scaler_visual(
        size: int,
        text_version: str,
        max_act: float,
        x_positions: list[float],
        max_layer: int
    ):
        """
        Optionally add a visual neuron for the prediction scaler,
        pinned at the very first column (x_positions[0]).
        """
        model = GeneratorNeuron.model
        if model.config.scaler.target_is_unscaled:
            model.prediction_scaler_neuron = None
            return  # nothing to show
        print("Adding prediction scaler")
        # first column is where our scaler sits
        x_coord = x_positions[max_layer-1] - size

        y_coord = model.height // 2 - size // 2
        print(f"target scaler x_coord = {x_coord}")
        scaler_neuron = DisplayModel__NeuronScaler(
            model,
            left=x_coord+111 ,
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
        model.prediction_scaler_neuron = scaler_neuron




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
