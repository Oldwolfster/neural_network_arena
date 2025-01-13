from typing import List

from src.NeuroForge.DisplayClasses import DisplayModel, DisplayNeuron, DisplayConnection
from src.engine.Utils_DataClasses import ModelInfo


class ObjectPlacer:
    def __init__(self, screen, info: ModelInfo, canvas_width=400, canvas_height=300):
        self.info = info
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

        # Prepare the DisplayModel
        self.display_model = DisplayModel()
        self.display_model.id = info.model_id
        self.display_model.architecture = info.full_architecture
        self.display_model.layers = len(info.full_architecture)

        # Calculate positions and populate neurons/connections
        self.calculate_neuron_positions()
        self.calculate_connections()

    def calculate_neuron_positions(self):
        """Calculate neuron positions and populate DisplayNeuron objects."""
        layer_spacing = self.canvas_width // (len(self.info.full_architecture))  # Spacing for layers
        neuron_spacing = self.canvas_height // max(self.info.full_architecture[1:])  # Neurons per layer
        #print(f"Debugging in objectPlacer architecture = {self.info.full_architecture}")

                # Fixed size for neurons, adjust as needed
        neuron_width = 12
        neuron_height = 12

        for layer_index, neuron_count in enumerate(self.info.full_architecture[1:], start=1):  # Skip input layer
            layer_x = layer_spacing * layer_index
            for neuron_index in range(neuron_count):
                # Create DisplayNeuron
                neuron = DisplayNeuron(nid=neuron_index)
                neuron.location_left = layer_x
                neuron.location_top = neuron_spacing * (neuron_index + 1)
                neuron.location_width = neuron_width
                neuron.location_height = neuron_height
                neuron.layer = layer_index
                #print(f"Debugging in objectPlacer location_left = {layer_x}\tlocation_top={neuron.location_top}")

                # Append to the DisplayModel's neurons list
                self.display_model.neurons.append(neuron)

    def calculate_connections(self):
        """
        Generate DisplayConnection objects and add them to the DisplayModel.
        """
        # Group neurons by layer
        neurons_by_layer = {}
        for neuron in self.display_model.neurons:
            if neuron.layer not in neurons_by_layer:
                neurons_by_layer[neuron.layer] = []
            neurons_by_layer[neuron.layer].append(neuron)

        # Generate connections between adjacent layers
        sorted_layers = sorted(neurons_by_layer.keys())
        for i in range(len(sorted_layers) - 1):
            current_layer_neurons = neurons_by_layer[sorted_layers[i]]
            next_layer_neurons = neurons_by_layer[sorted_layers[i + 1]]

            for from_neuron in current_layer_neurons:
                for to_neuron in next_layer_neurons:
                    # Create a DisplayConnection
                    connection = DisplayConnection(from_neuron, to_neuron)
                    self.display_model.connections.append(connection)


    def get_display_model(self):
        """Return the populated DisplayModel."""
        return self.display_model


#def create_display_models(model_info_list: List[ModelInfo]) -> List[DisplayModel]:
    """Create and return a list of DisplayModel instances."""
#    return [ObjectPlacer(info).get_display_model() for info in model_info_list]


