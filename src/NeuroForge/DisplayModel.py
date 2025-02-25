import re

import pygame
from src.NeuroForge import Const
from src.NeuroForge.EZSurface import EZSurface
from src.NeuroForge.GeneratorNeuron import GeneratorNeuron
from src.engine.ModelConfig import ModelConfig


class DisplayModel(EZSurface):
    __slots__ = ("config", "neurons", "connections")
    def __init__(self, config: ModelConfig, position: dict):
        """Initialize a display model using pixel-based positioning."""
        super().__init__(
            width_pct=0, height_pct=0, left_pct=0, top_pct=0,  # Ignore percent-based positioning
            pixel_adjust_width=position["width"],
            pixel_adjust_height=position["height"],
            pixel_adjust_left=position["left"],
            pixel_adjust_top=position["top"],
            bg_color=Const.COLOR_FOR_BACKGROUND
        )
        self.config = config
        self.neurons = [[] for _ in range(len(self.config.architecture))]  # Nested list by layers
        self.connections = []  # List of neuron connections

    def initialize_with_model_info(self):
        """Create neurons and connections based on architecture."""
        GeneratorNeuron.create_neurons(self)
        #self.create_arrows(True)  # Forward pass arrows

    def render(self):
        """Draw neurons and connections."""
        self.clear()
        self.draw_model_name()  # Draw model name in top-right corner
        self.draw_border()
#        for connection in self.connections:
#            connection.draw_connection(self.surface)
        for layer in self.neurons:
            for neuron in layer:
                neuron.draw_neuron(self.surface)

    def update_me(self):
        """Update neurons based on the latest iteration data."""
        for layer in self.neurons:
            for neuron in layer:
                pass
                #neuron.update_neuron(Const.dm.db, iteration, epoch, self.model_id)
                #neuron.update_neuron(iteration, epoch, self.model_id)

    def draw_border(self):
        """Draw a rectangle around the perimeter of the display model."""
        pygame.draw.rect(
            self.surface, Const.COLOR_FOR_NEURON_BODY,
            (0, 0, self.width, self.height), 3
        )

    def draw_model_name(self):
        """Draw the model's name in the top-right corner of the model area."""
        font = pygame.font.Font(None, 36)
        text_surface = font.render(self.beautify_text(self.config.gladiator_name), True, Const.COLOR_BLACK)
        text_x = self.width - text_surface.get_width() - 10  # Align to right with margin
        text_y = 5  # Small margin from the top
        self.surface.blit(text_surface, (text_x, text_y))

    def beautify_text(self, text: str) -> str:
        """Replaces underscores with spaces and adds spaces before CamelCase words."""
        text = text.replace("_", " ")
        text = re.sub(r'(?<!^)(?=[A-Z])', ' ', text)
        return text