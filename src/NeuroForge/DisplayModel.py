import re

import pygame
from src.NeuroForge import Const
from src.NeuroForge.EZSurface import EZSurface
from src.NeuroForge.GeneratorNeuron import GeneratorNeuron
from src.engine.ModelConfig import ModelConfig


class DisplayModel(EZSurface):
    __slots__ = ("config", "neurons", "connections", "model_id")
    def __init__(self, config: ModelConfig, position: dict )   :
        """Initialize a display model using pixel-based positioning."""
        super().__init__(
            width_pct=0, height_pct=0, left_pct=0, top_pct=0,  # Ignore percent-based positioning
            pixel_adjust_width=position["width"],
            pixel_adjust_height=position["height"],
            pixel_adjust_left=position["left"],
            pixel_adjust_top=position["top"],
            bg_color=Const.COLOR_FOR_BACKGROUND
        )
        self.config         = config
        self.model_id       = config.gladiator_name
        self.neurons        = [[] for _ in range(len(self.config.architecture))]  # Nested list by layers
        self.connections    = []  # List of neuron connections

    def initialize_with_model_info(self):
        """Create neurons and connections based on architecture."""
        max_activation = self.get_max_activation_for_model(self.model_id)
        GeneratorNeuron.create_neurons(self, max_activation)
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
                neuron.draw_neuron()

    def update_me(self):
        for layer in self.neurons:
            for neuron in layer:
                neuron.update_neuron()

    def draw_border(self):
        """Draw a rectangle around the perimeter of the display model."""
        pygame.draw.rect(
            self.surface, Const.COLOR_FOR_NEURON_BODY,
            (0, 0, self.width, self.height), 3
        )

    def draw_model_name(self):
        """Draw the model's name in the top-right corner of the model area."""
        font = pygame.font.Font(None, 28)
        text_surface = font.render(self.beautify_text(self.config.gladiator_name), True, Const.COLOR_BLACK)
        text_x = self.width - text_surface.get_width() - 10  # Align to right with margin

        text_y = 5  # Small margin from the top
        self.surface.blit(text_surface, (text_x, text_y))

    def beautify_text(self, text: str) -> str:
        """Replaces underscores with spaces and adds spaces before CamelCase words."""
        text = text.replace("_", " ")
        text = re.sub(r'(?<!^)(?=[A-Z])', ' ', text)
        return text


    def get_max_activation_for_model(self,  model_id: str):
        """
        Retrieves the highest absolute activation value across all epochs and iterations for the given model.

        :param model_id: The model identifier
        :return: The maximum absolute activation value in the run
        """

        SQL_MAX_ACTIVATION = """
            SELECT MAX(abs_activation) AS max_activation
            FROM (
                SELECT ABS(activation_value) AS abs_activation
                FROM Neuron
                WHERE model = ?
                ORDER BY abs_activation ASC
                LIMIT (SELECT CAST(COUNT(*) * 0.95 AS INT) 
                       FROM Neuron WHERE model = ?)
            ) AS FilteredActivations;
        """

        result = self.config.db.query(SQL_MAX_ACTIVATION, (model_id, model_id))
        #print(f"Max activation for run {result}")
        # Return the max activation or a default value to prevent division by zero
        return result[0]['max_activation'] if result and result[0]['max_activation'] is not None else 1.0
