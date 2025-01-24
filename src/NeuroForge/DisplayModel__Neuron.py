import json
from ast import literal_eval
from typing import List
import pygame

from src.NeuroForge import mgr
from src.NeuroForge.EZPrint import EZPrint
from src.engine.RamDB import RamDB
from src.engine.Utils import smart_format
from src.NeuroForge.mgr import * # Imports everything into the local namespace


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
        # Create EZPrint instance
        self.ez_printer = EZPrint(mgr.font, color=(0, 0, 0), max_width=200, max_height=100, sentinel_char="\n")

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
        """
        Update this neuron based on its layer type (input or hidden).
        """
        if self.layer == 0:  # Input layer neuron
            self.update_input_neuron()
        else:  # Hidden or output layer neuron
            self.update_hidden_neuron(db, iteration, epoch, model_id)

    def update_input_neuron(self):
        """
        Update an input neuron with its corresponding value.
        """
        if self.position < len(DisplayModel__Neuron.input_values):
            value = DisplayModel__Neuron.input_values[self.position]
            self.weight_text = f"Input: {value:.3f}"

    def update_hidden_neuron(self, db: RamDB, iteration: int, epoch: int, model_id: str):
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
        if rs:
            self.weight_text = self.neuron_report_build_prediction_logic(rs[0])
        else:
            self.weight_text = ""
            #print(f"Query result: {rs}")
            #print(f"PREDICTIONS: {self.weight_text}")

    def draw_neuron(self, screen):
        # Define colors
        body_color = (0, 0, 255)  # Blue for the neuron body
        label_color = (70, 130, 180)  # Steel blue for the label strip
        text_color = (255, 255, 255)  # White for text on the label

        # Font setup
        font = pygame.font.Font(None, 18)
        if self.nid>=0:
            label_text = font.render(f"{self.label} (ID: {self.nid})", True, text_color)
        else:
            label_text = font.render(f"{self.label} (Input #{self.position + 1})", True, text_color)

        # Calculate label strip height based on text height
        text_height = label_text.get_height()
        label_strip_height = text_height + 8  # Add padding (e.g., 8 pixels for breathing room)

        # Draw the label strip
        pygame.draw.rect(
            screen,
            label_color,
            (self.location_left, self.location_top, self.location_width, label_strip_height)
        )

        # Draw the label text in the label strip
        screen.blit(label_text, (self.location_left + 5, self.location_top + (label_strip_height - text_height) // 2))

        # Draw the neuron rectangle (body)
        body_y_start = self.location_top + label_strip_height
        body_height = self.location_height - label_strip_height
        pygame.draw.rect(
            screen,
            body_color,
            (self.location_left, body_y_start, self.location_width, body_height),
            3  # Border width
        )

        # Render the neuron content below the label strip
        body_text_y_start = body_y_start + 5
        self.ez_printer.render(
            screen,
            text=self.weight_text,
            x=self.location_left + 5,
            y=body_text_y_start
        )

    def neuron_report_build_prediction_logic(self,row):
        """
        Build prediction logic for a single neuron (row).
        Loops through weights and inputs, generating labeled calculations.
        """
        nid = row.get('nid')  # Get neuron ID
        weights = json.loads(row.get('weights_before', '[]'))  # Deserialize weights
        inputs = json.loads(row.get('inputs', '[]'))  # Deserialize inputs

        # Generate prediction logic
        predictions = []
        self.raw_sum = 0
        print(f"In DisplayModel__Neuron INPUTS:{inputs}")
        print(f"In DisplayModel__Neuron WEIGHTS:{weights}")
        #if len(weights) == len(inputs):         # Validate lengths of weights and inputs
        for i, (w, inp) in enumerate(zip(weights, inputs), start=1):
            #label = f"W{i}I{i}"  # Update label to match new specs
            linesum= (w*inp)
            calculation = f"{smart_format(w)} * {smart_format(inp)} = {smart_format(w * inp)}"
            predictions.append(calculation)
            self.raw_sum += linesum
        return f"{'\n'.join(predictions)}\n{self.format_bias_and_raw_sum(row)}"

    def format_bias_and_raw_sum(self, row):
        """
        Format the bias and raw sum for display.

        Args:
            row (dict): A dictionary representing a single neuron instance.

        Returns:
            str: Nicely formatted string with bias and raw sum.
        """
        # Extract bias and raw sum from the dictionary
        bias = row.get('bias', 0)
        self.raw_sum +=bias

        # Format the values
        bias_str = f"Bias: {smart_format(bias)}"
        raw_sum_str = f"Raw Sum (z): {smart_format(self.raw_sum)}"

        # Combine them into a single string
        return f"{bias_str}\n{raw_sum_str}"



