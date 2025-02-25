import json
from ast import literal_eval
from typing import List
import pygame
from src.engine.ActivationFunction import get_activation_derivative_formula
from src.NeuroForge.DisplayModel__NeuronWeights import DisplayModel__NeuronWeights
from src.NeuroForge.EZPrint import EZPrint
from src.engine.RamDB import RamDB
from src.engine.Utils import smart_format, draw_gradient_rect
from src.NeuroForge import Const

class DisplayModel__Neuron:
    input_values = []   # Class variable to store inputs

    def __init__(self, nid: int, layer: int, position: int, output_layer: int, text_version: str,  model_id: str, screen: pygame.surface):
        self.model_id = model_id
        self.screen = screen
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

        self.banner_text = ""
        self.tooltip_columns = []
        self.weight_adjustments = ""
        self.error_signal_calcs = ""
        self.avg_err_sig_for_epoch = 0.0
        self.loss_gradient = 0.0

        # Conditional visualizer
        self.neuron_visualizer = DisplayModel__NeuronWeights(self)
        self.neuron_build_text = "fix me"
        #self.neuron_build_text = self.neuron_build_text_large if text_version == "Verbose" else self.neuron_build_text_small
        self.ez_printer = EZPrint(pygame.font.Font(None, 24), color=Const.COLOR_BLACK, max_width=200, max_height=100, sentinel_char="\n")

    def draw_neuron(self):
        """Draw the neuron visualization."""
        # Define colors
        self.update_neuron() #Get latest info for neuron

        #TODO add Gradient body_color = self.get_color_gradient(self.avg_err_sig_for_epoch, mgr.max_error)

        # Font setup
        font = pygame.font.Font(None, 24)

        # Banner text
        label_surface = font.render(f"ID: {self.label}", True, Const.COLOR_FOR_NEURON_TEXT)
        output_surface = font.render(self.activation_function, True, Const.COLOR_FOR_NEURON_TEXT)
        label_strip_height = label_surface.get_height() + 8  # Padding

        # Draw neuron banner
        banner_rect = pygame.Rect(self.location_left, self.location_top + 4, self.location_width, label_strip_height)
        draw_gradient_rect(self.screen, banner_rect, Const.COLOR_FOR_BANNER_START, Const.COLOR_FOR_BANNER_END)

        self.screen.blit(label_surface, (self.location_left + 5, self.location_top + (label_strip_height - label_surface.get_height()) // 2))
        right_x = self.location_left + self.location_width - output_surface.get_width() - 5
        self.screen.blit(output_surface, (right_x, self.location_top + (label_strip_height - output_surface.get_height()) // 2))

        # Draw the neuron body below the label
        body_y_start = self.location_top + label_strip_height
        body_height = self.location_height - label_strip_height
        pygame.draw.rect(
            self.screen,  Const.COLOR_FOR_NEURON_BODY,
            (self.location_left, body_y_start, self.location_width, body_height),
            border_radius=6, width=5
        )

        # Render visual elements
        if hasattr(self, 'neuron_visualizer') and self.neuron_visualizer:
            self.neuron_visualizer.render(self.screen, self, body_y_start)

    def update_neuron(self):
            if not self.update_avg_error():
                return #no record found so exit early
            # Parameterized query with placeholders
            SQL =   """
                SELECT  *
                FROM    Iteration I
                JOIN    Neuron N
                ON      I.model_id  = N.model 
                AND     I.epoch     = N.epoch_n
                AND     I.iteration = N.iteration_n
                WHERE   model = ? AND iteration_n = ? AND epoch_n = ? AND nid = ?
                ORDER BY epoch, iteration, model, nid 
            """


            params = (self.model_id, Const.CUR_ITERATION, Const.CUR_EPOCH, self.nid)
            # print(f"SQL in update_me: {SQL}")
            # print(f"Params: {params}")

            rs = self.db.query(SQL, params) # Execute query
            try:
                self.weight_text = self.neuron_build_text(rs[0])
                self.loss_gradient =  float(rs[0].get("loss_gradient", 0.0))
                self.error_signal_calcs = rs[0].get("error_signal_calcs")
                #print(f"calcsforerror{self.error_signal_calcs}")
                self.banner_text = f"{self.label}  Output: {smart_format( self.activation_value)}"
                #print(f"Query result: {rs}")
                #print(f"PREDICTIONS: {self.weight_text}")
            except:
                pass

    def update_avg_error(self):
        SQL = """
        SELECT AVG(ABS(error_signal)) AS avg_error_signal            
        FROM Neuron
        WHERE 
        model   = ? and
        epoch_n = ? and  -- Replace with the current epoch(ChatGPT is trolling us)
        nid     = ?      
        """
        params = (self.model_id,  Const.CUR_EPOCH, self.nid)
        rs = self.db.query(SQL, params)  # Execute query

        # ✅ Check if `rs` is empty before accessing `rs[0]`
        if not rs:
            #print("in update_avg_error returning false")
            return False  # No results found

        # ✅ Ensure `None` does not cause an error
        self.avg_err_sig_for_epoch = float(rs[0].get("avg_error_signal") or 0.0)
        #print("in update_avg_error returning TRUE")
        return True



