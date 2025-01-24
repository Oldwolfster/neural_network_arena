import json
from typing import List
import pygame

from src.NeuroForge.DisplayModel__Connection import DisplayConnection
from src.NeuroForge.DisplayModel__Neuron import DisplayNeuron
from src.NeuroForge.EZSurface import EZSurface
from src.engine.RamDB import RamDB


class DisplayModel(EZSurface):
    def __init__(self, screen, width_pct=60, height_pct=80, left_pct=10, top_pct=10):
        super().__init__(screen, width_pct, height_pct, left_pct, top_pct, bg_color=(240, 240, 240))
        self.neurons = []
        self.connections = []
        self.model_id = None

    def initialize_with_model_info(self, model_info):
        self.model_id = model_info.model_id
        # Create a modified architecture that excludes the input layer
        modified_architecture = model_info.full_architecture[1:]

        # Calculate layer and neuron spacing
        layer_spacing = self.width // len(modified_architecture)
        neuron_size, vertical_spacing = self.calculate_dynamic_neuron_layout(modified_architecture, self.height)

        self.neurons = []
        for layer_index, neuron_count in enumerate(modified_architecture):  # Skip input layer
            layer_neurons = []

            if len(modified_architecture) == 1:
                layer_spacing = self.width // 2  # Center it horizontally

            # Horizontal position for the current layer
            layer_x = layer_spacing * layer_index + 20

            # Calculate vertical centering
            total_layer_height = (neuron_size + vertical_spacing) * neuron_count - vertical_spacing
            vertical_offset = (self.height - total_layer_height) // 2

            for neuron_index in range(neuron_count):
                neuron = DisplayNeuron(nid=f"{layer_index}-{neuron_index}")
                neuron.layer = layer_index

                # Set position and size
                neuron.location_left = layer_x
                neuron.location_top = vertical_offset + (neuron_size + vertical_spacing) * neuron_index
                neuron.location_width = neuron_size
                neuron.location_height = neuron_size

                layer_neurons.append(neuron)
            self.neurons.append(layer_neurons)


        # Create connections
        self.connections = []
        for layer_index in range(len(modified_architecture) - 1):
            current_layer = self.neurons[layer_index]
            next_layer = self.neurons[layer_index + 1]
            for from_neuron in current_layer:
                for to_neuron in next_layer:
                    connection = DisplayConnection(from_neuron=from_neuron, to_neuron=to_neuron)
                    self.connections.append(connection)

    def render(self):
        """Draw neurons and connections on the model's surface."""
        self.clear()  # Clear the surface before rendering
        for connection in self.connections:
            connection.draw_me(self.surface)
        for layer in self.neurons:  # Iterate through each layer
            for neuron in layer:  # Iterate through each neuron in the layer
                neuron.draw_me(self.surface)

    def update_me(self, db: RamDB, iteration : int, epoch : int, model_id: str):
        print(f"Neuron structure: {self.neurons}")

        for layer in self.neurons:  # Iterate through each layer
            for neuron in layer:  # Iterate through each neuron in the layer
                neuron.update_me(db, iteration, epoch, self.model_id)

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
