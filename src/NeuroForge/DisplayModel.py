import pygame
from typing import List

from src.NeuroForge.DisplayModel__Neuron import DisplayModel__Neuron
from src.NeuroForge.DisplayModel__Connection import DisplayModel__Connection
from src.NeuroForge.EZSurface import EZSurface
from src.engine.RamDB import RamDB


class DisplayModel(EZSurface):
    def __init__(self, screen, data_labels, width_pct, height_pct, left_pct, top_pct, architecture=None):
        #print(f"IN DISPLAYMODEL -- left_pct = {left_pct}")
        super().__init__(screen, width_pct, height_pct, left_pct, top_pct, bg_color=(222, 222, 222))
        self.neurons = [[] for _ in range(len(architecture))] if architecture else []  # Nested list by layers
        self.connections = []  # List of connections between neurons
        self.model_id = None
        self.data_labels = data_labels
        self.architecture = architecture or []  # Architecture to define layers and neurons

    def initialize_with_model_info(self, model_info):
        """
        Populate neurons and connections based on the provided model information.
        """
        self.model_id = model_info.model_id
        self.architecture = model_info.full_architecture

        # Calculate layer and neuron spacing
        layer_spacing = self.width // len(self.architecture)
        neuron_size, vertical_spacing = self.calculate_dynamic_neuron_layout(self.architecture, self.height)

        # Create neurons
        self.neurons = []
        nid = -1
        for layer_index, neuron_count in enumerate(self.architecture):
            if layer_index == 0:
                continue
            layer_neurons = []

            # Horizontal position for the current layer
            layer_x = layer_spacing * layer_index - neuron_size / 2

            # Calculate vertical centering
            total_layer_height = (neuron_size + vertical_spacing) * neuron_count - vertical_spacing
            vertical_offset = (self.height - total_layer_height) // 2
            for neuron_index in range(neuron_count):
                if layer_index > 0:
                    nid += 1        #increment nid for all neurons that are not inputs
                neuron = DisplayModel__Neuron(nid=nid , layer=layer_index, position=neuron_index)


                # Set position and size
                neuron.location_left = layer_x
                neuron.location_top = vertical_offset + (neuron_size + vertical_spacing) * neuron_index
                neuron.location_width = neuron_size
                neuron.location_height = neuron_size

                layer_neurons.append(neuron)
            self.neurons.append(layer_neurons)

        # Create connections
        self.connections = []
        for layer_index in range(1, len(self.architecture) - 1):  # Start from the first hidden layer
            current_layer = self.neurons[layer_index - 1]  # Adjust to skip the input layer
            next_layer = self.neurons[layer_index]
            for from_neuron in current_layer:
                for to_neuron in next_layer:
                    connection = DisplayModel__Connection(from_neuron=from_neuron, to_neuron=to_neuron)
                    self.connections.append(connection)

        # *** Add Input-to-First-Hidden-Layer Connections ***
        self.add_input_connections()

    def add_input_connections(self):
        """
        Creates connections from a single fixed point on the left edge of the model area
        to the first hidden layer neurons.
        """
        if not self.neurons or len(self.neurons) == 0:
            return  # No hidden layers exist

        first_hidden_layer = self.neurons[0]  # First hidden layer
        if not first_hidden_layer:
            return  # Edge case: No neurons in first hidden layer

        # 🔹 Fixed single origin point (Middle of the left edge of grey box)
        origin_x = self.left  # Left edge of the grey box
        origin_y = self.top + self.height // 2  # Vertical center of the grey box

        # Create a virtual input source **at a single fixed point**
        virtual_input = DisplayModel__Neuron(nid=-1, layer=-1, position=0)
        virtual_input.location_left = origin_x  # Fixed at left boundary
        virtual_input.location_top = origin_y  # Fixed vertical middle

        for neuron in first_hidden_layer:
            # Connect from the single origin point to each neuron in the first hidden layer
            self.connections.append(DisplayModel__Connection(from_neuron=virtual_input, to_neuron=neuron))

    def render(self):
        """
        Draw neurons and connections on the model's surface.
        """
        self.clear()  # Clear the surface before rendering


        # Draw connections
        for connection in self.connections:
            connection.draw_connection(self.surface)

        # Draw neurons
        for layer in self.neurons:
            for neuron in layer:
                neuron.draw_neuron(self.surface)

        # Draw the input box last
        if hasattr(self, "input_box"):
            self.input_box.render()

    def update_me(self, db: RamDB, iteration: int, epoch: int, model_id: str):
        """
        Update neuron and connection information based on the current state in the database.
        """
        DisplayModel__Neuron.retrieve_inputs(db, iteration, epoch, model_id)
        for layer in self.neurons:
            for neuron in layer:
                neuron.update_neuron(db, iteration, epoch, self.model_id)

        # (Optional) If connections have dynamic properties, update them too
        for connection in self.connections:
            connection.update_connection()

    def calculate_dynamic_neuron_layout(self, architecture, surface_height, margin=1, max_neuron_size=1500, spacing_ratio=0.03):
        """
        Calculate neuron size and spacing to fit within the surface height.

        Parameters:
            architecture (list[int]): Number of neurons in each layer.
            surface_height (int): Height of the surface in pixels.
            margin (int): Margin around the edges of the surface.
            max_neuron_size (int): Maximum size of a single neuron.
            spacing_ratio (float): Ratio of spacing to neuron size (e.g., 0.5 means spacing is half the size).

        Returns:
            tuple: (neuron_size, vertical_spacing)
        """
        max_neurons = max(architecture)
        available_height = surface_height - (2 * margin)  # Deduct margins

        # Calculate tentative neuron size
        tentative_neuron_size = available_height // (max_neurons + (max_neurons - 1) * spacing_ratio)

        # Clamp neuron size to the maximum allowed
        neuron_size = min(tentative_neuron_size, max_neuron_size)

        # Calculate spacing based on the neuron size
        vertical_spacing = int(neuron_size * spacing_ratio)

        return neuron_size, vertical_spacing
