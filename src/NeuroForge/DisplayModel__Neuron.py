import json
from ast import literal_eval
from typing import List
import pygame

from src.engine.ActivationFunction import get_activation_derivative_formula
from src.neuroForge_original import mgr
from src.neuroForge_original.DisplayModel__NeuronText import DisplayModel__NeuronText
from src.neuroForge_original.DisplayModel__NeuronWeights import DisplayModel__NeuronWeights
from src.neuroForge_original.EZPrint import EZPrint
from src.engine.RamDB import RamDB
from src.engine.Utils import smart_format, draw_gradient_rect
from src.neuroForge_original.mgr import * # Imports everything into the local namespace
from src.NeuroForge import Const

class DisplayModel__Neuron:
    input_values = []   # Class variable to store inputs

    def __init__(self, nid: int, layer: int, position: int, output_layer: int, text_version: str,  model_id: str):
        self.model_id = model_id
        self.db = Const.dm.db
        self.nid = nid
        self.layer = layer
        self.position = position
        self.output_layer = output_layer
        self.label = f"{layer}-{position}"

        # Positioning
        self.location_left = 0
        self.location_top = 0
        self.location_width = 0
        self.location_height = 0

        # Neural properties
        self.weights = []
        self.neuron_inputs = []
        self.bias = 0
        self.raw_sum = 0
        self.activation_function = ""
        self.activation_value = 0
        self.activation_gradient = 0

        # Visualization properties
        self.weight_text = ""
        self.banner_text = ""
        self.tooltip_columns = []
        self.weight_adjustments = ""
        self.error_signal_calcs = ""
        self.avg_err_sig_for_epoch = 0.0
        self.loss_gradient = 0.0

        # Conditional visualizer
        # self.neuron_visualizer = DisplayModel__NeuronText(self)
        self.neuron_visualizer = DisplayModel__NeuronWeights(self, self.model_id)
        self.neuron_build_text = "fix me"
        #self.neuron_build_text = self.neuron_build_text_large if text_version == "Verbose" else self.neuron_build_text_small
        self.ez_printer = EZPrint(pygame.font.Font(None, 24), color=Const.COLOR_BLACK, max_width=200, max_height=100, sentinel_char="\n")

    def draw_neuron(self, screen):
        """Draw the neuron visualization."""
        # Define colors
        body_color = Const.COLOR_FOR_NEURON_BODY
        text_color = Const.COLOR_WHITE

        #TODO add Gradient body_color = self.get_color_gradient(self.avg_err_sig_for_epoch, mgr.max_error)

        # Font setup
        font = pygame.font.Font(None, 24)

        # Banner text
        label_surface = font.render(f"ID: {self.label}", True, text_color)
        output_surface = font.render(self.activation_function, True, text_color)
        label_strip_height = label_surface.get_height() + 8  # Padding

        # Draw neuron banner
        banner_rect = pygame.Rect(self.location_left, self.location_top + 4, self.location_width, label_strip_height)
        draw_gradient_rect(screen, banner_rect, Const.COLOR_FOR_BANNER_START, Const.COLOR_FOR_BANNER_END)

        screen.blit(label_surface, (self.location_left + 5, self.location_top + (label_strip_height - label_surface.get_height()) // 2))
        right_x = self.location_left + self.location_width - output_surface.get_width() - 5
        screen.blit(output_surface, (right_x, self.location_top + (label_strip_height - output_surface.get_height()) // 2))

        # Draw the neuron body below the label
        body_y_start = self.location_top + label_strip_height
        body_height = self.location_height - label_strip_height
        pygame.draw.rect(
            screen, body_color,
            (self.location_left, body_y_start, self.location_width, body_height),
            border_radius=6, width=5
        )

        # Render visual elements
        if hasattr(self, 'neuron_visualizer') and self.neuron_visualizer:
            self.neuron_visualizer.render(screen, self.ez_printer, body_y_start, self.weight_text, self.location_left)
