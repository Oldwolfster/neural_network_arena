import pygame
from src.NeuroForge import Const
from src.NeuroForge.DisplayModel__ConnectionForward import DisplayModel__ConnectionForward
from src.NeuroForge.DisplayModel__Neuron import DisplayModel__Neuron
from src.NeuroForge.EZSurface import EZSurface
from src.NeuroForge.ModelGenerator import ModelGenerator


class DisplayModel(EZSurface):
    def __init__(self, model_id, architecture):
        """Initialize a display model using pixel-based positioning."""
        position = ModelGenerator.model_positions[model_id]  # Get calculated layout position

        super().__init__(
            width_pct=0, height_pct=0, left_pct=0, top_pct=0,  # Ignore percent-based positioning
            pixel_adjust_width=position["width"],
            pixel_adjust_height=position["height"],
            pixel_adjust_left=position["left"],
            pixel_adjust_top=position["top"],
            bg_color=Const.COLOR_WHITE
        )

        self.model_id = model_id
        self.architecture = architecture
        self.neurons = [[] for _ in range(len(architecture))]  # Nested list by layers
        self.connections = []  # List of neuron connections


    def initialize_with_model_info(self):
        """Create neurons and connections based on architecture."""
        self.create_neurons()
        self.create_arrows(True)  # Forward pass arrows

    def position_neurons(self, margin=20, gap=60, max_neuron_size=400):
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

        print(f"✅ Positioned {nid + 1} neurons for model {self.model_id}.")





    def create_arrows(self, forward: bool):
        """Create neuron connections."""
        self.connections = []
        for layer_index in range(1, len(self.architecture) - 1):
            current_layer = self.neurons[layer_index - 1]
            next_layer = self.neurons[layer_index]
            for weight_index, from_neuron in enumerate(current_layer):
                for to_neuron in next_layer:
                    connection = DisplayModel__ConnectionForward(
                        from_neuron=from_neuron, to_neuron=to_neuron, weight_index=weight_index
                    )
                    self.connections.append(connection)
        self.add_input_connections(forward)
        self.add_output_connections(forward)

    def add_input_connections(self, forward: bool):
        """Create connections from the left edge to the first hidden layer."""
        first_hidden_layer = self.neurons[0]
        if not first_hidden_layer:
            return
        origin_point = (0, self.height // 2)
        for neuron in first_hidden_layer:
            self.connections.append(DisplayModel__ConnectionForward(from_neuron=origin_point, to_neuron=neuron))

    def add_output_connections(self, forward: bool):
        """Create connections from last output neuron to prediction box."""
        dest_point = (self.width, 144)
        self.connections.append(DisplayModel__ConnectionForward(from_neuron=self.neurons[-1][0], to_neuron=dest_point))

    def render(self):
        """Draw neurons and connections."""
        self.clear()
        for connection in self.connections:
            connection.draw_connection(self.surface)
        for layer in self.neurons:
            for neuron in layer:
                neuron.draw_neuron(self.surface)

    def update_me(self, iteration: int, epoch: int):
        """Update neurons based on the latest iteration data."""
        for layer in self.neurons:
            for neuron in layer:
                neuron.update_neuron(Const.dm.db, iteration, epoch, self.model_id)
