from typing import List

from src.engine.Utils_DataClasses import ModelInfo


class ObjectPlacer:
    def __init__(self, info: ModelInfo, canvas_width=400, canvas_height=300):
        self.info = info
        self.neuron_positions = {}  # To store neuron positions, e.g., {layer: [(x, y), ...]}
        self.connections = []  # To store connections, e.g., [((x1, y1), (x2, y2)), ...]
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

        # Perform initial calculations
        self.calculate_neuron_positions()
        self.calculate_connections()

        # Debug: Print architecture and positions
        print(f"ModelVisualization constructor: {info.model_id}, Architecture: {info.full_architecture}")
        print(f"Neuron positions: {self.neuron_positions}")
        print(f"Connections: {self.connections}")

    def calculate_neuron_positions(self):
        """
        Calculate positions for neurons in each layer based on the architecture,
        skipping the input layer.
        """
        # Calculate spacing for layers and neurons
        layer_spacing = self.canvas_width // (len(self.info.full_architecture))  # No +1 since we skip the input layer
        neuron_spacing = self.canvas_height // max(self.info.full_architecture[1:])  # Skip input layer for spacing

        # Iterate over the architecture starting from the second element (index 1)
        for layer_index, neuron_count in enumerate(self.info.full_architecture[1:], start=1):  # Start at layer 1
            layer_x = layer_spacing * layer_index  # Position layers
            self.neuron_positions[layer_index] = [
                (layer_x, neuron_spacing * (i + 1)) for i in range(neuron_count)
            ]


    def calculate_connections(self):
        """
        Generate connection positions between neurons in adjacent layers.
        """
        self.connections = []  # Clear existing connections
        for layer_index in range(len(self.neuron_positions) - 1):
            current_layer = self.neuron_positions[layer_index]
            next_layer = self.neuron_positions[layer_index + 1]
            for neuron_start in current_layer:
                for neuron_end in next_layer:
                    self.connections.append((neuron_start, neuron_end))


def populate_model_info( model_info_list: List[ModelInfo]) -> List[ObjectPlacer]:
    rendering_details_per_model = []
    for info in model_info_list:
        single_model_rendering_details= ObjectPlacer(info)
        rendering_details_per_model.append(single_model_rendering_details)
    return rendering_details_per_model