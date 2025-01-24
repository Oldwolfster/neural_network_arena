import pygame
from typing import List
from src.NeuroForge.DisplayModel__Neuron import DisplayModel__Neuron
from src.NeuroForge.DisplayModel__Connection import DisplayModel__Connection
from src.NeuroForge.EZSurface import EZSurface
from src.engine.RamDB import RamDB


class DisplayModel(EZSurface):
    def __init__(self, screen, width_pct, height_pct, left_pct, top_pct, architecture=None):
        #print(f"IN DISPLAYMODEL -- left_pct = {left_pct}")
        super().__init__(screen, width_pct, height_pct, left_pct, top_pct, bg_color=(240, 240, 240))
        self.neurons = [[] for _ in range(len(architecture))] if architecture else []  # Nested list by layers
        self.connections = []  # List of connections between neurons
        self.model_id = None
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
            layer_neurons = []

            # Horizontal position for the current layer
            layer_x = layer_spacing * layer_index + 20

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
        for layer_index in range(len(self.architecture) - 1):
            current_layer = self.neurons[layer_index]
            next_layer = self.neurons[layer_index + 1]
            for from_neuron in current_layer:
                for to_neuron in next_layer:
                    connection = DisplayModel__Connection(from_neuron=from_neuron, to_neuron=to_neuron)
                    self.connections.append(connection)

    def render(self):
        """
        Draw neurons and connections on the model's surface.
        """
        self.clear()  # Clear the surface before rendering

        # Draw connections first (to avoid overlapping neurons)
        for connection in self.connections:
            connection.draw_connection(self.surface)

        # Draw neurons
        for layer in self.neurons:
            for neuron in layer:
                neuron.draw_neuron(self.surface)

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

    def calculate_dynamic_neuron_layout(self, architecture, surface_height, margin=5, max_neuron_size=1500, spacing_ratio=0.25):
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
