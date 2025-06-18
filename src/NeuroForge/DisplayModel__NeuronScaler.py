import pygame
from src.NNA.Legos.ActivationFunctions import get_activation_derivative_formula
from src.NNA.Legos.Optimizers import *
from src.NeuroForge.DisplayModel__NeuronScalerThresholder import DisplayModel__NeuronScalerThresholder

from src.NeuroForge.DisplayModel__NeuronWeights import DisplayModel__NeuronWeights
from src.NeuroForge.EZPrint import EZPrint
from src.NNA.engine.Neuron import Neuron
from src.NNA.engine.UtilsPyGame import smart_format, draw_gradient_rect, is_numeric
from src.NeuroForge import Const
import json

from src.NNA.engine.Utils_DataClasses import ez_debug

from src.NeuroForge.DisplayModel__Neuron_Base import DisplayModel__Neuron_Base
from src.NeuroForge.DisplayModel__NeuronScalerInputs import  DisplayModel__NeuronScalerInputs
from src.NeuroForge.DisplayModel__NeuronScalerPrediction import DisplayModel__NeuronScalerPrediction
from src.NNA.engine.Utils_DataClasses import ez_debug

class DisplayModel__NeuronScaler(DisplayModel__Neuron_Base):
    def _from_base_constructor(self):
        """Called from DisplayModel_Neuron_Base constructor"""
        #ez_debug(text_ver = self.text_version)
        # Decrease width if possible
        target_width= 136.9
        if self.location_width> target_width:   #room to thin it out
            if not  self.is_input:      #only move tthe prediction.. leav input where it is
                self.location_left += (self.location_width-target_width) * .5
            self.location_width = target_width
        self.location_height = 161.696
        #self.location_top= 96
        if self.is_input == True:
            self.banner_text = "Input Scaler"
            self.neuron_visualizer      = DisplayModel__NeuronScalerInputs(self, self.ez_printer)
            #ez_debug(banner=self.banner_text,inorout=self.is_input)
        elif self.nid == -2 : #-2 means thresholder
            self.neuron_visualizer      = DisplayModel__NeuronScalerThresholder(self, self.ez_printer)
            self.banner_text = "Thresholder"
        # Either way, show prediction window  ##
        elif 1==1:
            self.neuron_visualizer      = DisplayModel__NeuronScalerPrediction(self, self.ez_printer)
            #if self.model.layer_width < 24: #Based on layer
            if self.neuron_visualizer.neuron.location_width < 24:
                self.banner_text = "Prediction"
            else:
                self.banner_text = "Prediction"
            #ez_debug(banner=self.banner_text,inorout=self.is_input)

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

        body_rect = (self.location_left, body_y_start, self.location_width, body_height)
        #pygame.draw.rect(self.screen,  Const.COLOR_FOR_NEURON_BODY, (self.location_left, body_y_start, self.location_width, body_height), border_radius=6, width=7)
        pygame.draw.rect(self.screen,  Const.COLOR_FOR_NEURON_BODY, body_rect, border_radius=6, width=7)
        #print(f"drawing scaler neuron at = {body_rect}")


        # Render visual elements
        if hasattr(self, 'neuron_visualizer') and self.neuron_visualizer:
            #ez_debug(In_Visualizer_renderCall=self.neuron_visualizer)
            self.neuron_visualizer.render() #, self, body_y_start)


        # Draw neuron banner
        banner_rect = pygame.Rect(self.location_left, self.location_top + 4, self.location_width, label_strip_height)
        draw_gradient_rect(self.screen, banner_rect, Const.COLOR_FOR_BANNER_START, Const.COLOR_FOR_BANNER_END)
        self.screen.blit(label_surface, (self.location_left + 5, self.location_top + 5 + (label_strip_height - label_surface.get_height()) // 2))
        right_x = self.location_left + self.location_width - output_surface.get_width() - 5
        self.screen.blit(output_surface, (right_x, self.location_top + 5 + (label_strip_height - output_surface.get_height()) // 2))

