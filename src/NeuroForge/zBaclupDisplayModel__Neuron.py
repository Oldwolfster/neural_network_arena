import json
from ast import literal_eval
from typing import List
import pygame

from src.neuroForge import mgr
from src.neuroForge.EZPrint import EZPrint
from src.engine.RamDB import RamDB
from src.engine.Utils import smart_format, draw_gradient_rect
from src.neuroForge.mgr import * # Imports everything into the local namespace


class DisplayModel__Neuron:
    input_values = []   # Class variable to store inputs
    def __init__(self, nid:int, layer: int, position: int):
        #print(f"Instantiating neuron Pnid={nid}\tlabel={label}")
        self.location_left=0
        self.location_top=0
        self.location_width=0
        self.location_height = 0
        self.nid = nid
        self.layer = layer
        self.position = position
        self.label = f"{layer}-{position}" #need to define, try to use existing standard
        self.weights = []
        self.bias = 0
        self.weight_count = []
        self.weight_formula_txt = ""
        self.raw_sum = 0
        self.activation_function = ""
        self.activation_value =0
        self.weight_text = ""
        self.banner_text = ""
        self.mouse_x = 0
        self.mouse_y = 0
        # Create EZPrint instance
        self.ez_printer = EZPrint(pygame.font.Font(None, 24)
                                  , color=(0, 0, 0), max_width=200, max_height=100, sentinel_char="\n")
    def is_hovered(self, offset_x : int, offset_y : int): #"""Check if the mouse is over this neuron."""
        mouse_x, mouse_y = pygame.mouse.get_pos()
        self.mouse_x =  mouse_x-offset_x
        self.mouse_y = mouse_y - offset_y  #print(f"left={self.location_left}\twidth={self.location_left + self.location_width  }\tmouse={mouse_x}tmouse2={self.mouse_x}")
        return (self.location_left <= self.mouse_x <= self.location_left + self.location_width and self.location_top <= self.mouse_y <= self.location_top + self.location_height)



    @classmethod
    def retrieve_inputs(cls, db: RamDB, iteration: int, epoch: int, modelID: str):
        """
        Retrieve inputs from the database and store in the class variable.
        """
        sql = """  
            SELECT * FROM Iteration 
            WHERE epoch = ? AND iteration = ?  
        """
        params = (epoch, iteration)

        # Execute query
        rs = db.query(sql, params)

        # Parse and store inputs
        if rs:
            raw_inputs = rs[0].get("inputs", cls.input_values)
            try:
                cls.input_values = json.loads(raw_inputs)
            except json.JSONDecodeError:
                cls.input_values = literal_eval(raw_inputs)

    def update_neuron(self, db: RamDB, iteration: int, epoch: int, model_id: str):
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
        params = (model_id, iteration, epoch, self.nid)
        # print(f"SQL in update_me: {SQL}")
        # print(f"Params: {params}")

        rs = db.query(SQL, params) # Execute query
        self.weight_text = self.neuron_report_build_prediction_logic(rs[0])
        self.banner_text = f"{self.label}  Output: {smart_format( self.activation_value)}"
        #print(f"Query result: {rs}")
        #print(f"PREDICTIONS: {self.weight_text}")

    def draw_neuron(self, screen):
        # Define colors
        body_color = (0, 0, 255)  # Blue for the neuron body
        #label_color = (70, 130, 180)  # Steel blue for the label strip
        text_color = (255, 255, 255)  # White for text on the label

        # Font setup
        font = pygame.font.Font(None, 24)

        # **Split banner text into two parts:**
        label_text = f"ID: {self.label}"  # Left side (Neuron ID)
        output_text = f"{smart_format(self.activation_value)}"  # Right side

        # Render both texts separately
        label_surface = font.render(label_text, True, text_color)
        output_surface = font.render(output_text, True, text_color)

        # Get text dimensions
        text_height = label_surface.get_height()
        label_strip_height = text_height + 8  # Padding (8px)

        # Draw the banner
        #pygame.draw.rect(
        #    screen,
        #    label_color,
        #    (self.location_left, self.location_top, self.location_width, label_strip_height)
        #)


        # Draw the neuron body below the label
        body_y_start = self.location_top + label_strip_height
        body_height = self.location_height - label_strip_height
        pygame.draw.rect(
            screen,
            body_color,
            (self.location_left, body_y_start, self.location_width, body_height),
            border_radius=6,
            width= 3  # Border width

        )

        # Draw neuron banner
        banner_rect = pygame.Rect(self.location_left, self.location_top+4, self.location_width, label_strip_height)
        draw_gradient_rect(screen, banner_rect, (70, 130, 180), (25, 25, 112))
        text_height -= 6
        screen.blit(label_surface, (self.location_left + 5, self.location_top + (label_strip_height - text_height) // 2)) # **Blit Label on the Left**
        right_x = self.location_left + self.location_width - output_surface.get_width() - 5  # Align to right  # **Blit Output on the Right**
        screen.blit(output_surface, (right_x, self.location_top + (label_strip_height - text_height) // 2))

        # Render neuron details inside the body
        body_text_y_start = body_y_start + 5
        self.ez_printer.render(
            screen,
            text=self.weight_text,
            x=self.location_left + 11,
            y=body_text_y_start + 7
        )

    def neuron_report_build_prediction_logic(self, row):
        """
        Generate a formatted report for a single neuron.
        Includes weighted sum calculations, bias, activation details,
        and backpropagation details (activation gradient, error signal).
        """
        prediction_logic = self.build_prediction_logic(row)
        bias_activation_info = self.format_bias_and_activation(row)
        backprop_details = self.format_backpropagation_details(row)  # 🔥 New Function!
        #print(row)
        weight_adjustments =  row.get('weight_adjustments')
        return f"{prediction_logic}\n{bias_activation_info}\n{backprop_details}\n{weight_adjustments}"

    # ---------------------- Existing Functions ---------------------- #

    def build_prediction_logic(self, row):
        """
        Compute weighted sum calculations and format them.
        """
        nid = row.get('nid')  # Get neuron ID
        weights = json.loads(row.get('weights_before', '[]'))  # Deserialize weights
        inputs = json.loads(row.get('neuron_inputs', '[]'))  # Deserialize inputs

        # Generate weighted sum calculations
        predictions = []
        self.raw_sum = 0

        for i, (w, inp) in enumerate(zip(weights, inputs), start=1):
            linesum = (w * inp)
            calculation = f"{smart_format(inp)} * {smart_format(w)} = {smart_format(linesum)}"
            predictions.append(calculation)
            self.raw_sum += linesum  # Accumulate weighted sum

        return '\n'.join(predictions)

    def format_bias_and_activation(self, row):
        """
        Format the bias, raw sum, and activation function for display.
        """
        bias = row.get('bias_before', 0)
        self.raw_sum += bias

        # Activation function details
        activation_name = row.get('activation_name', 'Unknown')
        self.activation_value = row.get('activation_value', None)        #THE OUTPUT
        activation_gradient = row.get('activation_gradient', None)  # From neuron

        # Format strings
        bias_str = f"Bias: {smart_format(bias)}"
        raw_sum_str = f"Raw Sum: {smart_format(self.raw_sum)}"
        #activation_str = f"{activation_name}: {smart_format(activation_value)}" if activation_value is not None else ""
        activation_str = f"{activation_name} Gradient: {smart_format(activation_gradient)}"
        return f"{bias_str}\n{raw_sum_str}\n{activation_str}"

    # ---------------------- 🔥 New Function! 🔥 ---------------------- #

    def format_backpropagation_details(self, row):
        """
        Format and display backpropagation details:
        - Activation Gradient (A')
        - Error Signal (δ)
        """

        error_signal = row.get('error_signal', None)  # From neuron


        error_signal = f"Error Signal (δ): {smart_format(error_signal)}"

        return f"{error_signal}"


    def render_tooltip(self, screen):   #"""Render the tooltip with neuron details."""
        tooltip_width = 250
        tooltip_height = 140
        tooltip_x = self.mouse_x + 10
        tooltip_y = self.mouse_y + 10

        # Ensure tooltip doesn't go off screen
        if tooltip_x + tooltip_width > screen.get_width():
            tooltip_x -= tooltip_width + 20
        if tooltip_y + tooltip_height > screen.get_height():
            tooltip_y -= tooltip_height + 20

        # Draw background box
        pygame.draw.rect(screen, (255, 255, 200), (tooltip_x, tooltip_y, tooltip_width, tooltip_height))
        pygame.draw.rect(screen, (0, 0, 0), (tooltip_x, tooltip_y, tooltip_width, tooltip_height), 2)

        # Font setup
        font = pygame.font.Font(None, 22)

        # Define tooltip content
        info_lines = [
            f"{self.label}",
            f"Activation: {self.activation_value:.4f}",
            f"Raw Sum: {self.raw_sum:.4f}",
            f"Bias: {self.bias:.4f}",
        ]

        # Render text inside tooltip
        for i, text in enumerate(info_lines):
            label = font.render(text, True, (255, 0, 0))
            screen.blit(label, (tooltip_x + 5, tooltip_y + 5 + i * 20))  # Draw text on screen
