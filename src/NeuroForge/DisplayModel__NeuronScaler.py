import pygame
from src.Legos.ActivationFunctions import get_activation_derivative_formula
from src.Legos.Optimizers import *

from src.NeuroForge.DisplayModel__NeuronWeights import DisplayModel__NeuronWeights
from src.NeuroForge.EZPrint import EZPrint
from src.engine.Neuron import Neuron
from src.engine.Utils import smart_format, draw_gradient_rect, is_numeric
from src.NeuroForge import Const
import json

from src.engine.Utils_DataClasses import ez_debug

from src.NeuroForge.DisplayModel__Neuron_Base import DisplayModel__Neuron_Base
from src.NeuroForge.DisplayModel__NeuronScalers import  DisplayModel__NeuronScalers
from src.engine.Utils_DataClasses import ez_debug

class DisplayModel__NeuronScaler(DisplayModel__Neuron_Base):
    def _from_base_constructor(self):
        """Called from DisplayModel_Neuron_Base constructor"""
        #ez_debug(text_ver = self.text_version)
        self.location_width *= .8
        self.neuron_visualizer      = DisplayModel__NeuronScalers(self, self.ez_printer)
        if self.text_version == "Verbose":
            self.banner_text = "Scaler"
        else:
            self.banner_text = "Input Scaler"

    def draw_neuron(self):
        """Draw the neuron visualization."""

        # Font setup
        font = pygame.font.Font(None, 30) #TODO remove and use EZ_Print

        #ez_debug(In_scaler_print=self.banner_text)
        # Banner text
        label_surface = font.render(self.banner_text, True, Const.COLOR_FOR_NEURON_TEXT)
        output_surface = font.render(self.activation_function, True, Const.COLOR_FOR_NEURON_TEXT)
        label_strip_height = label_surface.get_height() + 8  # Padding

        # Draw the neuron body below the label
        body_y_start = self.location_top + label_strip_height
        body_height = self.location_height - label_strip_height
        #pygame.draw.rect(self.screen,  Const.COLOR_FOR_NEURON_BODY, (self.location_left, body_y_start, self.location_width, body_height), border_radius=6, width=7)

        # Draw neuron banner
        #banner_rect = pygame.Rect(self.location_left, self.location_top + 4, self.location_width, label_strip_height)
        #draw_gradient_rect(self.screen, banner_rect, Const.COLOR_FOR_BANNER_START, Const.COLOR_FOR_BANNER_END)
        #self.screen.blit(label_surface, (self.location_left + 5, self.location_top + 5 + (label_strip_height - label_surface.get_height()) // 2))
        #right_x = self.location_left + self.location_width - output_surface.get_width() - 5
        #self.screen.blit(output_surface, (right_x, self.location_top + 5 + (label_strip_height - output_surface.get_height()) // 2))

        # Render visual elements
        if hasattr(self, 'neuron_visualizer') and self.neuron_visualizer:
            #ez_debug(In_Visualizer_renderCall=self.neuron_visualizer)
            self.neuron_visualizer.render() #, self, body_y_start)



